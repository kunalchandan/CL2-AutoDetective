# Remoting into Server

Install and launch VcXsrv
```
winget install vcxsrv

C:\ProgramData\Microsoft\Windows\Start Menu\Programs\VcXsrv
```
After launching xLaunch just go with all the defaults and click finish.


Set the vscode configuration for ssh

```
Ctrl + Shift + P

Remote SSH Open Configuration File
```

It should have this in there
```
Host 129.97.68.241
  HostName 129.97.68.241
  User e5_5044
  ForwardX11 yes
  ForwardX11Trusted yes
```

Or from the command line.

```
ssh -v -Y e5_5044@129.97.68.241
```

Ensure that you get something like this or better at the end.
```
debug1: Remote: /home/e5_5044/.ssh/authorized_keys:2: key options: agent-forwarding port-forwarding pty user-rc x11-forwarding
debug1: Remote: /home/e5_5044/.ssh/authorized_keys:2: key options: agent-forwarding port-forwarding pty user-rc x11-forwarding
debug1: No xauth program.
Warning: No xauth data; using fake authentication data for X11 forwarding.
debug1: Requesting X11 forwarding with authentication spoofing.
```

If this works, you can disregard the `-v` parameter.

Ensure that in the remote server after `ssh`ing into the server you can run.
```
xeyes
```

Ensure that you can see the eyes that pop up and that they can follow around your cursor.


I don't know if this is necessary but you might need.
```
export DISPLAY=localhost:10.0
```
