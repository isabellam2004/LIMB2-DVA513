### Dependencies
The eigen library found here: https://gitlab.com/libeigen/eigen

### Use
The library is cloned into a folder called `libs`, if you put it somewhere else you'll have to change it in `CMakeLists.txt`.

Then you should create a build folder and go into it:<br>
`mkdir build && cd build`

Run these commands:<br>
`cmake ..`<br
`cmake --build . --config Release`

Then run the app:<br>
`Release/limb_IK.exe`