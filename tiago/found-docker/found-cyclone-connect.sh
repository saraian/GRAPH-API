#!/bin/bash
# PAL-style CycloneDDS unicast to a physical TIAGo (same config as
# `pal connection start`, without an interactive subshell).
#
# Source it so exports apply:  . found-cyclone-connect
# Or run it to only write the XML: found-cyclone-connect
ROBOT_IP=${TIAGO_ROBOT_IP:-10.68.0.1}
IFACE=${TIAGO_NET_IFACE:-}
DOMAIN=${TIAGO_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-1}}
OUT=${CYCLONE_PEER_XML:-${HOME:-/home/user}/.ros/cyclone_tiago.xml}
TEMPLATE=${CYCLONE_TEMPLATE:-/opt/pal/alum/share/cyclone_dev_cfg/config/cyclone_config.xml}

_found_cyclone_fail() {
  echo "found-cyclone-connect: $*" >&2
  return 1 2>/dev/null || exit 1
}

if [ -z "$IFACE" ]; then
  IFACE=$(ip -o route get "$ROBOT_IP" 2>/dev/null | awk '{
    for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }
  }')
fi
[ -n "$IFACE" ] || _found_cyclone_fail "cannot find interface toward $ROBOT_IP"
[ -f "$TEMPLATE" ] || _found_cyclone_fail "missing Cyclone template $TEMPLATE"

mkdir -p "$(dirname "$OUT")"

python3 - "$TEMPLATE" "$OUT" "$ROBOT_IP" "$IFACE" "$DOMAIN" <<'PY' || _found_cyclone_fail "failed to write Cyclone XML"
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

template, out, peer, iface, domain = sys.argv[1:6]
tree = ET.parse(template)
root = tree.getroot()
domain_tag = root.find("./Domain")
if domain_tag is None:
    raise SystemExit("Cyclone XML: missing Domain")
domain_tag.set("Id", str(domain))
interfaces = root.find("./Domain/General/Interfaces")
if interfaces is None:
    raise SystemExit("Cyclone XML: missing Interfaces")
# Dual-homed hosts (robot USB Ethernet + internet/LAN): advertise ONLY the
# robot NIC. Keeping `lo` makes Cyclone publish 127.0.0.1 as a data locator;
# the robot then tries localhost for large samples (images) and the viewer
# stays on "CONNECTING TO LIVE ROS STREAM" while small topics still work.
for child in list(interfaces):
    interfaces.remove(child)
interfaces.append(ET.Element("NetworkInterface", {"name": iface, "priority": "1"}))
# Unicast-only (AllowMulticast=false) + a single robot peer means local
# FOUND nodes never discover each other, so /object_descriptions never
# reaches the object manager. Peer this machine's robot-NIC address too;
# do not add lo / 127.0.0.1 (those locators break the camera).
local_ip = None
try:
    ip_out = subprocess.check_output(
        ["ip", "-4", "-o", "addr", "show", "dev", iface], text=True)
    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", ip_out)
    if m:
        local_ip = m.group(1)
except Exception:
    local_ip = None
peers = root.find("./Domain/Discovery/Peers")
if peers is None:
    raise SystemExit("Cyclone XML: missing Peers")
wanted = [peer]
if local_ip and local_ip not in wanted:
    wanted.append(local_ip)
for child in list(peers):
    addr = child.attrib.get("Address", "")
    if addr in wanted or addr in ("localhost", "127.0.0.1"):
        peers.remove(child)
for addr in reversed(wanted):
    peers.insert(0, ET.Element("Peer", {"Address": addr}))
tree.write(out, encoding="utf-8", xml_declaration=True)
print(out)
PY

unset ROS_LOCALHOST_ONLY || true
export ROS_LOCALHOST_ONLY=
export ROS_DOMAIN_ID="$DOMAIN"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="$OUT"
export PAL_ROBOT_CONNECTED=1
export TIAGO_ROBOT_IP="$ROBOT_IP"
export TIAGO_NET_IFACE="$IFACE"
export TIAGO_ROS_DOMAIN_ID="$DOMAIN"

echo "CycloneDDS peer $ROBOT_IP on $IFACE (ROS_DOMAIN_ID=$DOMAIN)"
echo "CYCLONEDDS_URI=$CYCLONEDDS_URI"
