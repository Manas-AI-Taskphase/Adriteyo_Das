#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <queue>
#include <vector>
#include <cmath>

struct Node {
    int x, y;
    double cost;
    double heuristic;
    Node* parent;

    Node(int x_, int y_, double cost_, double heuristic_, Node* parent_) :
            x(x_), y(y_), cost(cost_), heuristic(heuristic_), parent(parent_) {}
};

std::vector<Node*> AStar(Node* start, Node* goal, const std::vector<std::vector<int>>& map, int width, int height) {
    std::vector<Node*> path;

    std::priority_queue<Node*, std::vector<Node*>,
            decltype([](Node* a, Node* b) { return a->cost + a->heuristic > b->cost + b->heuristic; })> open;

    start->cost = 0;
    start->heuristic = std::sqrt(std::pow(goal->x - start->x, 2) + std::pow(goal->y - start->y, 2));
    open.push(start);

    while (!open.empty()) {
        Node* current = open.top();
        open.pop();

        if (current->x == goal->x && current->y == goal->y) {
            while (current != nullptr) {
                path.push_back(current);
                current = current->parent;
            }
            break;
        }

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;

                int nx = current->x + dx;
                int ny = current->y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height && map[ny][nx] == 0) {
                    double new_cost = current->cost + std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));
                    double heuristic = std::sqrt(std::pow(goal->x - nx, 2) + std::pow(goal->y - ny, 2));

                    bool found = false;
                    for (Node* node : open) {
                        if (node->x == nx && node->y == ny) {
                            found = true;
                            if (node->cost > new_cost) {
                                node->cost = new_cost;
                                node->parent = current;
                            }
                            break;
                        }
                    }

                    if (!found) {
                        Node* neighbor = new Node(nx, ny, new_cost, heuristic, current);
                        open.push(neighbor);
                    }
                }
            }
        }
    }

    return path;
}

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    int width = msg->info.width;
    int height = msg->info.height;
    double resolution = msg->info.resolution;

    std::vector<std::vector<int>> map(height, std::vector<int>(width));

    // Populate the map with data from the message
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            map[y][x] = msg->data[index];
        }
    }

    Node* start = new Node(10, 10, 0, 0, nullptr);
    Node* goal = new Node(50, 50, 0, 0, nullptr);

    std::vector<Node*> path = AStar(start, goal, map, width, height);

    ros::NodeHandle nh;
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("path", 1);
    nav_msgs::Path path_msg;
    path_msg.header.frame_id = "map";

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = (*it)->x * resolution;
        pose.pose.position.y = (*it)->y * resolution;
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);
    }

    path_pub.publish(path_msg);
}

int main(int argc, char** argv) {
    // Initialize ROS
    ros::init(argc, argv, "path_planner_node");
    ros::NodeHandle nh;

    ros::Subscriber map_sub = nh.subscribe("map", 1, mapCallback);

    ros::spin();

    return 0;
}
