#!/usr/bin/env bash
# CAN interface initialization for Linux (socketcan + gs_usb candleLight adapters).
#
# With a single adapter plugged in: brings up `can0` at 1 Mbps.
# With multiple adapters: binds each one to can0, can1, ... in USB enumeration
# order so a multi-Piper rig can be distinguished.
#
# Run once after every boot / USB replug.
#
# Usage:
#   bash scripts/setup_can.sh                   # all adapters, 1 Mbps
#   bash scripts/setup_can.sh 1000000           # explicit bitrate
#   bash scripts/setup_can.sh 1000000 can0      # only bring up can0

set -e

BITRATE="${1:-1000000}"
ONLY_IFACE="${2:-}"

# --- Step 1: load gs_usb kernel module ---
if ! lsmod | grep -q "^gs_usb"; then
    echo "Loading gs_usb kernel module..."
    sudo modprobe gs_usb
fi

# --- Step 2: find every candleLight adapter (1d50:606f) ---
mapfile -t DEVICES < <(
    grep -l "1d50" /sys/bus/usb/devices/*/idVendor 2>/dev/null | while read -r f; do
        d=$(dirname "$f")
        if grep -q "606f" "$d/idProduct" 2>/dev/null; then
            basename "$d"
        fi
    done
)

if [ ${#DEVICES[@]} -eq 0 ]; then
    echo "ERROR: no candleLight USB adapter (1d50:606f) found. Is it plugged in?"
    exit 1
fi
echo "Found ${#DEVICES[@]} candleLight adapter(s): ${DEVICES[*]}"

# --- Step 3: bind every adapter so the kernel creates can0, can1, ... ---
for DEV in "${DEVICES[@]}"; do
    BIND_IFACE="${DEV}:1.0"
    if [ ! -e "/sys/bus/usb/drivers/gs_usb/${BIND_IFACE}" ]; then
        echo "Binding ${BIND_IFACE} to gs_usb..."
        sudo sh -c "echo '${BIND_IFACE}' > /sys/bus/usb/drivers/gs_usb/bind"
        sleep 0.3
    fi
done

# --- Step 4: configure + bring up every canN that now exists ---
mapfile -t CAN_IFACES < <(ip -o link show | awk -F': ' '/link\/can/{print $2}' | sort)

if [ ${#CAN_IFACES[@]} -eq 0 ]; then
    echo "ERROR: no canN interfaces appeared after binding. Check dmesg."
    exit 1
fi

for IFACE in "${CAN_IFACES[@]}"; do
    if [ -n "$ONLY_IFACE" ] && [ "$IFACE" != "$ONLY_IFACE" ]; then
        continue
    fi
    echo "Configuring $IFACE @ ${BITRATE} bps"
    sudo ip link set "$IFACE" down 2>/dev/null || true
    sudo ip link set "$IFACE" type can bitrate "$BITRATE"
    sudo ip link set "$IFACE" up
done

echo
echo "Up interfaces:"
ip -brief link show | awk '/can[0-9]+/'
echo
echo "Next: 'uv run --no-sync --active scripts/piper_identify_arms.py'"
echo "to confirm which canN corresponds to the arm you want to use."
