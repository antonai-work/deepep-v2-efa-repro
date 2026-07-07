#!/bin/bash
exec >/tmp/sshd_install.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q openssh-server 2>&1 | tail -3
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
# Use a shared known keypair (generated deterministically, ephemeral, pod-local). Generate on pod0, copy to pod1.
# sshd on a non-22 port to avoid any host conflict; passwordless root via authorized_keys
sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/; s/#\?PubkeyAuthentication.*/PubkeyAuthentication yes/; s/#\?Port .*/Port 2222/' /etc/ssh/sshd_config
grep -qE "^Port 2222" /etc/ssh/sshd_config || echo "Port 2222" >> /etc/ssh/sshd_config
/usr/sbin/sshd -D -p 2222 >/tmp/sshd.run.log 2>&1 &
sleep 1
echo "sshd_started_pid=$(pgrep -f 'sshd -D' | head -1)"
ss -tlnp 2>/dev/null | grep 2222 | head -1
echo "SSHD_INSTALL_DONE"
