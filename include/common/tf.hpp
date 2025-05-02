#ifndef TF2_HPP
#define TF2_HPP

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <string>
#include <opencv2/core.hpp>
#include <shared_mutex>
#include <unordered_set>

namespace tf2 {

class Quaternion {
public:
    float x, y, z, w;

    Quaternion() : x(0.0), y(0.0), z(0.0), w(1.0) {}
    Quaternion(float x_, float y_, float z_, float w_)
        : x(x_), y(y_), z(z_), w(w_) {}

    cv::Matx33d toRotationMatrix() const {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float xw = x * w, yw = y * w, zw = z * w;

        return cv::Matx33d(
            1 - 2 * (yy + zz), 2 * (xy - zw),     2 * (xz + yw),
            2 * (xy + zw),     1 - 2 * (xx + zz), 2 * (yz - xw),
            2 * (xz - yw),     2 * (yz + xw),     1 - 2 * (xx + yy)
        );
    }
    void setRPY(double roll, double pitch, double yaw) {
        double cy = cos(yaw * 0.5);
        double sy = sin(yaw * 0.5);
        double cp = cos(pitch * 0.5);
        double sp = sin(pitch * 0.5);
        double cr = cos(roll * 0.5);
        double sr = sin(roll * 0.5);

        w = cr * cp * cy + sr * sp * sy;
        x = sr * cp * cy - cr * sp * sy;
        y = cr * sp * cy + sr * cp * sy;
        z = cr * cp * sy - sr * sp * cy;
    }

};

class Matrix3x3 {
public:
    float m[3][3];

    Matrix3x3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    Matrix3x3(float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22) {
    m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
    m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
    m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }

    Matrix3x3(const Quaternion& q) {
        cv::Matx33d rot = q.toRotationMatrix();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = static_cast<float>(rot(i, j));
    }
    const float* operator[](int i) const { return m[i]; }
    float* operator[](int i) { return m[i]; }

    void getRotation(Quaternion& q) const {
        float trace = m[0][0] + m[1][1] + m[2][2];
        if (trace > 0.0) {
            float s = std::sqrt(trace + 1.0f) * 2.0f;
            q.w = 0.25f * s;
            q.x = (m[2][1] - m[1][2]) / s;
            q.y = (m[0][2] - m[2][0]) / s;
            q.z = (m[1][0] - m[0][1]) / s;
        } else {
            int i = (m[0][0] > m[1][1]) ? ((m[0][0] > m[2][2]) ? 0 : 2) : ((m[1][1] > m[2][2]) ? 1 : 2);
            float s;
            if (i == 0) {
                s = std::sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
                q.w = (m[2][1] - m[1][2]) / s;
                q.x = 0.25f * s;
                q.y = (m[0][1] + m[1][0]) / s;
                q.z = (m[0][2] + m[2][0]) / s;
            } else if (i == 1) {
                s = std::sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
                q.w = (m[0][2] - m[2][0]) / s;
                q.x = (m[0][1] + m[1][0]) / s;
                q.y = 0.25f * s;
                q.z = (m[1][2] + m[2][1]) / s;
            } else {
                s = std::sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
                q.w = (m[1][0] - m[0][1]) / s;
                q.x = (m[0][2] + m[2][0]) / s;
                q.y = (m[1][2] + m[2][1]) / s;
                q.z = 0.25f * s;
            }
        }
    }

    void getRPY(double& roll, double& pitch, double& yaw) const {
        if (std::abs(m[2][0]) >= 1.0) {
            pitch = m[2][0] < 0 ? M_PI / 2.0 : -M_PI / 2.0;
            roll = std::atan2(-m[0][1], -m[0][2]);
            yaw = 0.0;
        } else {
            pitch = std::asin(-m[2][0]);
            roll = std::atan2(m[2][1], m[2][2]);
            yaw = std::atan2(m[1][0], m[0][0]);
        }
    }
};

inline std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
    os << "Quaternion(x=" << q.x << ", y=" << q.y << ", z=" << q.z << ", w=" << q.w << ")";
    return os;
}



} // namespace tf2

struct Position {
    float x, y, z;
    Position() : x(0.0f), y(0.0f), z(0.0f) {}
    Position(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Transform {
    Position position;
    tf2::Quaternion orientation;
    double timestamp = 0.0;

    Transform() : position(), orientation(), timestamp(0.0) {}
    Transform(Position p, tf2::Quaternion q, double ts = 0.0) : position(p), orientation(q), timestamp(ts) {}

    cv::Matx44d toMatrix() const {
        tf2::Matrix3x3 R(orientation);
        cv::Matx44d T = cv::Matx44d::eye();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T(i, j) = R[i][j];
        T(0, 3) = position.x;
        T(1, 3) = position.y;
        T(2, 3) = position.z;
        return T;
    }

    static Transform fromMatrix(const cv::Matx44d& T) {
        Position pos(T(0, 3), T(1, 3), T(2, 3));
        tf2::Matrix3x3 R(
            T(0, 0), T(0, 1), T(0, 2),
            T(1, 0), T(1, 1), T(1, 2),
            T(2, 0), T(2, 1), T(2, 2)
        );
        tf2::Quaternion q;
        R.getRotation(q);
        return {pos, q};
    }

    static Transform compose(const Transform& a, const Transform& b) {
        cv::Matx44d result = a.toMatrix() * b.toMatrix();
        return fromMatrix(result);
    }
};

inline Transform createTf(float x, float y, float z, const tf2::Quaternion& q, double ts = 0.0) {
    return Transform(Position(x, y, z), q, ts);
}

inline tf2::Quaternion eulerToQuaternion(double roll, double pitch, double yaw) {
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    return q;
}

class TfTree {
public:
    void setTransform(const std::string& parent, const std::string& child, const Transform& tf) {
        std::unique_lock lock(mutex_);
        nodes_[child] = FrameNode{parent, tf};
    }

    bool getTransform(const std::string& source_frame, const std::string& target_frame, Transform& out) const {
        std::shared_lock lock(mutex_);
        if (source_frame == target_frame) {
            out = Transform(); // Identity
            return true;
        }

        try {
            std::vector<std::string> path = getPathToRoot(source_frame);
            std::vector<std::string> target_path = getPathToRoot(target_frame);

            // Find common ancestor
            int i = path.size() - 1, j = target_path.size() - 1;
            while (i >= 0 && j >= 0 && path[i] == target_path[j]) {
                --i;
                --j;
            }

            // Compose: source -> common -> target (invert)
            Transform tf_src_to_common = accumulatePath(path, i + 1);
            Transform tf_target_to_common = accumulatePath(target_path, j + 1);
            Transform tf_common_to_target = invert(tf_target_to_common);

            out = Transform::compose(tf_common_to_target, tf_src_to_common);
            return true;
        } catch (...) {
            return false;
        }
    }

    Transform transform(const Transform& input, const std::string& target_frame) const {
        Transform tf_map;
        if (!getTransform(input_frame(input), target_frame, tf_map)) {
            throw std::runtime_error("No transform from " + input_frame(input) + " to " + target_frame);
        }
        return Transform::compose(tf_map, input);
    }
    Transform transform(const Position& pos, const std::string& source_frame, const std::string& target_frame) const {
        // 获取当前转换关系
        Transform transform;
        if (!getTransform(source_frame, target_frame, transform)) {
            throw std::runtime_error("Cannot find transform from " + source_frame + " to " + target_frame);
        }
    
        // 将 Position 位置转化为矩阵
        cv::Matx44d mat_pos = transform.toMatrix();
        cv::Matx44d mat_input = cv::Matx44d::eye();
        mat_input(0, 3) = pos.x;
        mat_input(1, 3) = pos.y;
        mat_input(2, 3) = pos.z;
    
        // 应用变换
        cv::Matx44d mat_result = mat_pos * mat_input;
        Position result_pos(static_cast<float>(mat_result(0, 3)), static_cast<float>(mat_result(1, 3)), static_cast<float>(mat_result(2, 3)));
    
        // 返回包含转换后的 Position 的 Transform
        Transform result_transform = Transform::fromMatrix(mat_result);
        result_transform.position = result_pos;  // 更新 Transform 的位置部分
        return result_transform;
    }
    
    Transform transform(const tf2::Quaternion& ori, const std::string& source_frame, const std::string& target_frame) const {
        // 获取当前转换关系
        Transform transform;
        if (!getTransform(source_frame, target_frame, transform)) {
            throw std::runtime_error("Cannot find transform from " + source_frame + " to " + target_frame);
        }
    
        // 将 tf2::Quaternion 转换为矩阵
        cv::Matx44d mat_transform = transform.toMatrix();
        cv::Matx44d mat_input = cv::Matx44d::eye();
    
        // 旋转部分由传入的 Quaternion 表示
        tf2::Matrix3x3 rotation(ori);
        cv::Matx33d R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R(i, j) = rotation[i][j];
            }
        }
    
        // 更新变换矩阵
        mat_input.get_minor<3, 3>(0, 0) = R;
        cv::Matx44d mat_result = mat_transform * mat_input;
    
        // 提取新的方向
        tf2::Quaternion result_orientation;
        tf2::Matrix3x3 mat_result_rotation(
            mat_result(0, 0), mat_result(0, 1), mat_result(0, 2),
            mat_result(1, 0), mat_result(1, 1), mat_result(1, 2),
            mat_result(2, 0), mat_result(2, 1), mat_result(2, 2)
        );
        mat_result_rotation.getRotation(result_orientation);
    
        // 返回包含转换后的方向的 Transform
        Transform result_transform = Transform::fromMatrix(mat_result);
        result_transform.orientation = result_orientation;  // 更新 Transform 的方向部分
        return result_transform;
    }
    
    
    

private:
    struct FrameNode {
        std::string parent_frame;
        Transform transform;
    };

    std::unordered_map<std::string, FrameNode> nodes_;
    mutable std::shared_mutex mutex_;

    std::string input_frame(const Transform& t) const {
        for (const auto& [frame, node] : nodes_) {
            if (&node.transform == &t) return frame;
        }
        return "";
    }

    std::vector<std::string> getPathToRoot(const std::string& frame) const {
        std::vector<std::string> path;
        std::unordered_set<std::string> visited;

        std::string current = frame;
        while (nodes_.count(current)) {
            if (visited.count(current)) throw std::runtime_error("TF loop detected.");
            visited.insert(current);
            path.push_back(current);
            current = nodes_.at(current).parent_frame;
        }
        path.push_back(current);  // root (may not be in map)
        return path;
    }

    Transform accumulatePath(const std::vector<std::string>& path, int end_idx) const {
        Transform result;
        for (int i = 0; i < end_idx; ++i) {
            const auto& f = path[i];
            result = Transform::compose(nodes_.at(f).transform, result);
        }
        return result;
    }

    Transform invert(const Transform& tf) const {
        cv::Matx44d inv = tf.toMatrix().inv();
        return Transform::fromMatrix(inv);
    }
};
#endif // TF2_HPP
