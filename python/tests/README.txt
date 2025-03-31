By default OrangePi version of Ubuntu comes with Python 2.7 and 3.9. To update to 3.12, use pyenv as described here:
https://medium.com/@aashari/easy-to-follow-guide-of-how-to-install-pyenv-on-ubuntu-a3730af8d7f0

Previously had tried to build from source, but although this seemed to build okay, it didn't become system default or seem to be available from vscode:
https://www.howtogeek.com/install-latest-python-version-on-ubuntu/

To get spidev to work, first go to 
  OrangePi Start Menu > Applications > Settings > OrangePi Config
The choose 
  System > Hardware
For this project, enable the following:
  spi0-m2-cs0-spidev
  spi4-m2-cs0-spidev
  uart0-m2
Note that enabling the wrong spidev can cause the wifi to not work anymore (probably shared?)
Reboot

Check the available SPIs in xterm
  ls -l /dev/spi*
Should see
  crw------- 1 root root 153, 0 Mar 31 08:56 /dev/spidev0.0
  crw------- 1 root root 153, 1 Mar 31 08:56 /dev/spidev4.0
So only root has rw permissions

Have to add rw permissions for user:
  cd /etc/udev/rules.d/
  sudo nano 50-spi.rules
In this file add
  SUBSYSTEM=="spidev", GROUP="spi", MODE="0660" 
This will create a new group called spi that user orangepi will be added to in order to access SPI without root
Create new group called spi
  sudo groupadd spi 
And add the user (orangepi) to it
  sudo usermod -a -G spi $USER
Reboot

After selecting the Python interpreter (3.12), make sure to install spidev and OPi
pip install --upgrade pip
pip install spidev
pip install OPi.GPIO

