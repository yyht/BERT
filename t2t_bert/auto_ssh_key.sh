rm /root/.ssh/id_rsa* -f
ssh-keygen -t rsa -f /root/.ssh/id_rsa -N "" -q
for ip in 131 134
do
	sshpass -p123456 ssh-copy-id -o "StrictHostKeyChecking no" -i /root/.ssh/id_rsa.pub 192.168.3.$ip &>/dev/null
	if [ $? -eq 0];then
		action "fenfa 192.168.3.$ip" /bin/true
	else
		action "fenfa 192.168.3.$ip" /bin/false
	echo ""
done

