#include <iostream>
#include "opencv2/opencv.hpp"
#include "depthai/depthai.hpp"


int main() {
    dai::Pipeline pipeline;

    auto cam = pipeline.create<dai::node::Camera>();
    auto xout = pipeline.create<dai::node::XLinkOut>();

    xout->setStreamName("video");

    cam->video.link(xout->input);

    try {
        dai::Device device(pipeline);
    } catch(const std::exception& e) {
        std::cerr << "No device found: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}