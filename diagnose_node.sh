#!/bin/bash
# ===========================================================================
# diagnose_node.sh — 节点 / 文件系统健康自检
#
# 用途:当 cap-x 启动后所有进程卡死、kill -9 也杀不掉、感觉"节点卡死进不去"时,
# 跑这个脚本判断到底是 (a) 某个盘 ($HOME/$WORK/$FAST/$SCRATCH) 读写超时,
# 还是 (b) GPU/驱动问题,还是 (c) 节点其实是好的。
#
# 所有检查都用 `timeout` 包住,脚本本身绝不会卡死。
#   关键判读: 任何一行出现 "HANG(124)" = 那条 I/O 没在限时内返回 = 那个盘/设备卡了。
#
# 用法:   bash diagnose_node.sh            # 默认每项超时 10s
#         TMO=20 bash diagnose_node.sh     # 自定义超时秒数
#         PIDS="2883740 2883703" bash diagnose_node.sh   # 额外探测指定卡死进程
# ===========================================================================

TMO="${TMO:-10}"   # 每项检查的超时(秒)

# 带超时跑一条命令,统一打印 OK / HANG(124) / FAIL(rc)
run() {
    local label="$1"; shift
    local out rc
    out=$(timeout "$TMO" "$@" 2>&1); rc=$?
    if   [ "$rc" -eq 0   ]; then printf "  [OK]        %s\n" "$label"
    elif [ "$rc" -eq 124 ]; then printf "  [HANG(124)] %s  <-- 这个路径/设备卡住了!\n" "$label"
    else                         printf "  [FAIL(%s)]  %s  | %s\n" "$rc" "$label" "$(echo "$out" | head -1)"
    fi
}

# 对一个目录做 stat / ls / 写读删 三连
check_fs() {
    local name="$1" dir="$2"
    echo "----- $name = ${dir:-<未设置>} -----"
    if [ -z "$dir" ]; then echo "  (环境变量未设置,跳过)"; return; fi
    run "stat -f (查文件系统元数据)" stat -f "$dir"
    run "ls   (列目录)"             ls -1 "$dir"
    # 写 + sync + 读 + 删,放到一个子 shell 里整体计时
    local f="$dir/.capx_fscheck.$$"
    run "write+read+rm (实际读写)"  bash -c "echo ok > '$f' && sync && cat '$f' >/dev/null && rm -f '$f'"
}

echo "========================================================================"
echo " diagnose_node.sh   host=$(hostname)   time=$(date '+%F %T')   每项超时=${TMO}s"
echo "========================================================================"

# ---------------------------------------------------------------------------
echo
echo "### 1. 四大文件系统读写 (出现 HANG(124) 即为故障盘) ###"
check_fs HOME    "$HOME"
check_fs WORK    "$WORK"
check_fs FAST    "$FAST"
check_fs SCRATCH "$SCRATCH"

# ---------------------------------------------------------------------------
echo
echo "### 2. 关键库 / 模型路径 (能否真正读到内容,'连库都读不了'就卡在这) ###"
# 项目 venv 解释器
VENV_PY="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python"
run "读 venv python 头部字节        ($VENV_PY)"        bash -c "head -c 64 '$VENV_PY' >/dev/null"
# conda vllm 解释器(molmo 用)
CONDA_PY="$HOME/miniconda3/envs/vllm/bin/python3"
run "读 conda vllm python 头部字节  ($CONDA_PY)"       bash -c "head -c 64 '$CONDA_PY' >/dev/null"
# 模型缓存(在 scratch 上)
HF_HUB="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D33_222/zwang003/cache/huggingface}/hub"
run "ls 模型缓存目录               ($HF_HUB)"          ls -1 "$HF_HUB"
# 输出/日志目录
LOGD="${LOG_DIR:-/leonardo_scratch/fast/EUHPC_D33_222/cap-x/logs}"
run "ls 日志目录                   ($LOGD)"            ls -1 "$LOGD"

# ---------------------------------------------------------------------------
echo
echo "### 3. 挂载点全景 (df 本身卡住=有挂载死了) ###"
run "df -hT" df -hT
echo "  (上面若是 HANG(124),说明某个挂载点已经无响应)"

# ---------------------------------------------------------------------------
echo
echo "### 4. 节点资源 / GPU 健康 ###"
echo "  load / mem:"
run "uptime" uptime
run "free -h" free -h
echo "  GPU:"
run "nvidia-smi (显存/利用率)" nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

# ---------------------------------------------------------------------------
echo
echo "### 5. 内核日志里的存储/驱动报错 (需要权限,读不到可忽略) ###"
if dmesg -T >/dev/null 2>&1; then
    dmesg -T 2>/dev/null | grep -Ei 'lustre|gpfs|nfs|ll_|ptlrpc|timed out|timeout|evict|XID|Xid|nvrm|bad health' | tail -20
    echo "  (以上若有 'timed out / evicted / Xid' 等字样 = 存储或 GPU 驱动出问题)"
else
    echo "  (无权限读 dmesg,跳过)"
fi

# ---------------------------------------------------------------------------
echo
echo "### 6. 卡死进程探测 (可选: PIDS='pid1 pid2 ...' 传入) ###"
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        echo "  --- pid $pid ---"
        [ -d "/proc/$pid" ] || { echo "    (进程不存在)"; continue; }
        echo -n "    state : "; awk '/^State:/{print $2,$3}' "/proc/$pid/status" 2>/dev/null
        echo "    内核栈 (出现 lustre_/ll_/nfs_/ptlrpc 即卡在该盘 I/O;权限不足则空):"
        timeout 5 cat "/proc/$pid/stack" 2>/dev/null | head -8 | sed 's/^/      /'
        echo -n "    cwd   -> "; timeout 5 readlink "/proc/$pid/cwd" 2>&1
    done
else
    echo "  (未传 PIDS,跳过。例如: PIDS='2883740 2883703' bash diagnose_node.sh)"
fi

echo
echo "========================================================================"
echo " 判读速记:"
echo "   - 第1/2/3节出现 [HANG(124)]  => 那个盘读写超时 (你说的老问题复发)"
echo "   - 全部 [OK] 且 GPU 正常       => 节点/盘都健康,卡死另有原因"
echo "   - 第5节有 timed out/evicted/Xid => 存储或 GPU 驱动故障,需要换节点/报管理员"
echo "========================================================================"
