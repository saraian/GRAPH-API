#!/bin/bash
# Prepare a Linux development PC so PAL CycloneDDS unicast to a TIAGo works.
#
# Typical failure (this lab, 2026-09-07):
#   - ping/SSH to the robot succeed
#   - pal connection start opens a session
#   - ros2 topic list stays empty (/parameter_events, /rosout only)
# Cause: UFW default deny incoming drops robot -> PC UDP (DDS discovery).
# Secondary: strict rp_filter on a second NIC drops replies sourced from
# the robot's other addresses (e.g. wlan0 10.42.0.1).
#
# Usage:
#   ./tiago-host-dds.sh apply
#   ./tiago-host-dds.sh status
#   TIAGO_ROBOT_IP=10.68.0.1 TIAGO_NET_IFACE=enp6s0f3u2u2 ./tiago-host-dds.sh apply
#
# Env / flags:
#   TIAGO_ROBOT_IP     robot IPv4 (default 10.68.0.1)
#   TIAGO_NET_IFACE    NIC toward the robot (default: from `ip route get`)
#   --robot-ip / --interface override the same values
set -euo pipefail

DEFAULT_ROBOT_IP=10.68.0.1
DEFAULT_WLAN_IP=10.42.0.1
SYSCTL_FILE=/etc/sysctl.d/99-tiago-dds.conf
UFW_COMMENT="TIAGo CycloneDDS"

usage() {
  cat <<EOF
tiago-host-dds.sh {apply|status|dry-run} [--robot-ip IP] [--interface IFACE]

  apply     allow inbound UDP from the robot; relax rp_filter (needs sudo)
  status    show ping, route, UFW/firewalld, rp_filter (sudo if needed)
  dry-run   print what apply would do

Defaults: robot ${DEFAULT_ROBOT_IP}, interface from the route to that IP.
A graphical password prompt is used when sudo needs a password and DISPLAY
or WAYLAND_DISPLAY is set (kdialog, zenity, or pkexec).
EOF
  exit "${1:-1}"
}

ACTION=apply
ROBOT_IP=${TIAGO_ROBOT_IP:-$DEFAULT_ROBOT_IP}
WLAN_IP=${TIAGO_ROBOT_WLAN_IP:-$DEFAULT_WLAN_IP}
IFACE=${TIAGO_NET_IFACE:-}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    apply|status|dry-run) ACTION=$1; shift ;;
    --robot-ip) ROBOT_IP=${2:?}; shift 2 ;;
    --wlan-ip) WLAN_IP=${2:?}; shift 2 ;;
    --interface) IFACE=${2:?}; shift 2 ;;
    *) usage ;;
  esac
done

detect_iface() {
  local dev
  dev=$(ip -o route get "$ROBOT_IP" 2>/dev/null | awk '{
    for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }
  }')
  if [ -z "$dev" ] || [ "$dev" = "lo" ]; then
    echo "no IPv4 route to $ROBOT_IP; plug the robot NIC and set a $ROBOT_IP/24 address" >&2
    return 1
  fi
  printf '%s\n' "$dev"
}

if [ -z "$IFACE" ]; then
  IFACE=$(detect_iface)
fi

ASKPASS_FILE=

cleanup() {
  if [ -n "${ASKPASS_FILE:-}" ]; then
    rm -f "$ASKPASS_FILE"
  fi
}
trap cleanup EXIT

make_askpass() {
  ASKPASS_FILE=$(mktemp /tmp/tiago-dds-askpass.XXXXXX)
  if command -v kdialog >/dev/null 2>&1; then
    cat > "$ASKPASS_FILE" <<'EOF'
#!/bin/sh
exec kdialog --title "TIAGo DDS" --password "Administrator password to allow inbound UDP from the robot."
EOF
  elif command -v zenity >/dev/null 2>&1; then
    cat > "$ASKPASS_FILE" <<'EOF'
#!/bin/sh
exec zenity --password --title="TIAGo DDS" --text="Administrator password to allow inbound UDP from the robot."
EOF
  else
    rm -f "$ASKPASS_FILE"
    ASKPASS_FILE=
    return 1
  fi
  chmod 700 "$ASKPASS_FILE"
}

run_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && make_askpass; then
    SUDO_ASKPASS="$ASKPASS_FILE" sudo -A "$@"
    return
  fi
  if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v pkexec >/dev/null 2>&1; then
    pkexec "$@"
    return
  fi
  sudo "$@"
}

ufw_has_rule() {
  run_root ufw status 2>/dev/null | grep -qE "Anywhere on ${IFACE}[[:space:]]+ALLOW IN[[:space:]]+${ROBOT_IP}/udp"
}

firewalld_rule() {
  printf 'rule family="ipv4" source address="%s" protocol value="udp" accept\n' "$ROBOT_IP"
}

print_plan() {
  cat <<EOF
robot:      $ROBOT_IP
interface:  $IFACE
ping:       $(ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1 && echo ok || echo FAIL)
route:      $(ip -o route get "$ROBOT_IP" 2>/dev/null | tr -s ' ')
rp_filter:  all=$(sysctl -n net.ipv4.conf.all.rp_filter) ${IFACE}=$(sysctl -n "net.ipv4.conf.${IFACE}.rp_filter" 2>/dev/null || echo n/a)

apply would:
  1. UFW (if active): allow in on $IFACE from $ROBOT_IP and $WLAN_IP proto udp
  2. firewalld (if running): rich rule UDP from $ROBOT_IP
  3. sysctl: rp_filter=0 on all and $IFACE, persist in $SYSCTL_FILE
  4. ip route: $WLAN_IP/32 via $ROBOT_IP on $IFACE (robot Wi-Fi DDS locator)
EOF
}

apply_rules() {
  ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1 || {
    echo "warning: $ROBOT_IP did not answer ping on $IFACE; continuing" >&2
  }

  run_root bash -c "
set -euo pipefail
IFACE=$(printf '%q' "$IFACE")
ROBOT_IP=$(printf '%q' "$ROBOT_IP")
WLAN_IP=$(printf '%q' "$WLAN_IP")
SYSCTL_FILE=$(printf '%q' "$SYSCTL_FILE")
UFW_COMMENT=$(printf '%q' "$UFW_COMMENT")

if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q '^Status: active'; then
  for SRC in \"\$ROBOT_IP\" \"\$WLAN_IP\"; do
    [ -n \"\$SRC\" ] && [ \"\$SRC\" != none ] || continue
    if ufw status | grep -qE \"Anywhere on \${IFACE}[[:space:]]+ALLOW IN[[:space:]]+\${SRC}/udp\"; then
      echo \"UFW: rule already present for \${SRC} on \${IFACE}\"
    else
      ufw allow in on \"\$IFACE\" from \"\$SRC\" proto udp comment \"\$UFW_COMMENT\"
      echo \"UFW: allowed UDP from \${SRC} on \${IFACE}\"
    fi
  done
else
  echo 'UFW: not active (skipped)'
fi

if command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
  RULE='rule family=\"ipv4\" source address=\"'\$ROBOT_IP'\" protocol value=\"udp\" accept'
  if firewall-cmd --query-rich-rule=\"\$RULE\" >/dev/null 2>&1; then
    echo \"firewalld: rich rule already present for \${ROBOT_IP}\"
  else
    firewall-cmd --permanent --add-rich-rule=\"\$RULE\"
    firewall-cmd --reload
    echo \"firewalld: allowed UDP from \${ROBOT_IP}\"
  fi
fi

printf '%s\\n' \
  '# Multi-homed ROS 2 / PAL CycloneDDS: do not drop unicast DDS on the robot NIC.' \
  'net.ipv4.conf.all.rp_filter = 0' \
  \"net.ipv4.conf.\${IFACE}.rp_filter = 0\" \
  > \"\$SYSCTL_FILE\"
sysctl -w net.ipv4.conf.all.rp_filter=0 >/dev/null
sysctl -w \"net.ipv4.conf.\${IFACE}.rp_filter=0\" >/dev/null
echo \"rp_filter: 0 on all and \${IFACE} (saved \$SYSCTL_FILE)\"

# TIAGo Cyclone also advertises its Wi-Fi AP as a DDS locator. This PC has
# no route there, so writes to 10.42.0.1 fail. Send that /32 down the cable.
if [ -n \"\$WLAN_IP\" ] && [ \"\$WLAN_IP\" != none ]; then
  ip route replace \"\$WLAN_IP/32\" via \"\$ROBOT_IP\" dev \"\$IFACE\"
  echo \"route: \$WLAN_IP/32 via \$ROBOT_IP dev \$IFACE\"
fi
"
  echo
  echo "Host DDS path is open. In the PAL container run:"
  echo "  unset ROS_LOCALHOST_ONLY"
  echo "  pal connection start $ROBOT_IP $IFACE \${TIAGO_ROS_DOMAIN_ID:-1}"
}

show_status() {
  print_plan
  echo
  if command -v ufw >/dev/null; then
    echo "==== ufw ===="
    if run_root ufw status verbose 2>/dev/null; then
      :
    else
      echo "(need sudo for UFW status)"
    fi
  fi
  if command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1; then
    echo "==== firewalld ===="
    firewall-cmd --list-all 2>/dev/null | head -30 || true
  fi
}

case "$ACTION" in
  dry-run) print_plan ;;
  status) show_status ;;
  apply) apply_rules ;;
  *) usage ;;
esac
