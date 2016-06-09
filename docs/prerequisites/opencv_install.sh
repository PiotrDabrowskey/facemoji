# based on http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/
#!/usr/bin/env bash

# upgrade pre-installed packages
sudo apt-get update
sudo apt-get upgrade

# install tools used to compile OpenCV
sudo apt-get -y install build-essential cmake git pkg-config

# install libraries used to read image formats
sudo apt-get -y install libjpeg-dev libtiff-dev libjasper-dev libpng-dev

# install libraries used to read video formats
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

# install GTK for OpenCV's GUI feautures
sudo apt-get -y install libgtk2.0-dev

# install packages to optimize functions inside OpenCV, such as matrix operations
sudo apt-get -y install libatlas-base-dev gfortran

# install python3.5
sudo apt-get -y install python3.5-dev

# IMPORTANT:
# You need to install numpy to your Python environment.
# If you use, you can create new virtualenv with numpy installed:
conda create -n opencv numpy
# Next, you need to activate this virtualenv to compile OpenCV for the virtualenv's Python binaries
source activate opencv
# Upgrade pip
pip install --upgrade pip
# Install Pillow (Python Imaging Library)
pip install pillow

# clone OpenCV's repository
mkdir ~/opencv-install
cd ~/opencv-install
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.1.0
cd ~/opencv-install
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.1.0

# build OpenCV
mkdir ~/opencv-install/opencv/build
cd ~/opencv-install/opencv/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv-install/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..
# compile OpenCV (using 4 cpu cores)
make -j4

sudo make install
sudo ldconfig

# Create sym-link from site-packages of your environment to OpenCV Python bindings in:
# /usr/local/lib/python3.5/site-packages
# For conda virtualenv use your conda path and envs folder:
cd ~/anaconda3/envs/opencv/lib/python3.5/site-packages
ln -s /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so

# check if OpenCV is installed in currently selected environment
python -c "import cv2; print(cv2.__version__)"

# (optional) remove local repositories
cd ~
rm -r -f opencv-install

