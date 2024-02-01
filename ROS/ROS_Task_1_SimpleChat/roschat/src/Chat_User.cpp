#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>

ros::Publisher chat_pub;
std::string user_name;

void chatCallback(const std_msgs::String::ConstPtr & msg) {
  ROS_INFO("%s", msg -> data.c_str());
}

int main(int argc, char ** argv) {
  std::cout << "Enter your custom name: ";
  std::getline(std::cin, user_name);
  ros::init(argc, argv, user_name);
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(1);
  spinner.start();
  chat_pub = nh.advertise < std_msgs::String > ("chat", 1000);
  ros::Subscriber chat_sub = nh.subscribe("chat", 1000, chatCallback);

  while (ros::ok()) {
    std_msgs::String msg;
    std::string message;
    std::cout << user_name << ": ";
    std::getline(std::cin, message);
    msg.data = user_name + ": " + message;
    chat_pub.publish(msg);
    ros::spinOnce();
  }
  spinner.stop();
  return 0;
}
