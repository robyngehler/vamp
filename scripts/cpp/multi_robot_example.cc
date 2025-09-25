#include <iostream>

#include <Eigen/Geometry>

#include <vamp/collision/multi_robot.hh>
#include <vamp/robots/gp4.hh>

int main()
{
    using Robot = vamp::robots::GP4;
    constexpr std::size_t rake = vamp::FloatVectorWidth;

    auto environment = vamp::collision::Environment<vamp::FloatVector<rake>>{};

    auto config_a = Robot::template ConfigurationBlock<rake>::zero_vector();
    auto config_b = Robot::template ConfigurationBlock<rake>::zero_vector();

    const std::array<float, Robot::dimension> nominal = {0.f, 0.f, 0.f, -2.356f, 0.f, 0.f};
    for (std::size_t i = 0; i < Robot::dimension; ++i)
    {
        config_a[i] = config_a[i] + nominal[i];
        config_b[i] = config_b[i] + nominal[i];
    }

    config_a[0] = config_a[0] - 1.57F;
    config_b[0] = config_b[0] + 1.57F;

    Eigen::Isometry3f base_a = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f base_b = Eigen::Isometry3f::Identity();
    base_b.translate(Eigen::Vector3f(1.2F, 0.0F, 0.0F));

    auto state_a = vamp::collision::make_multi_robot_state<Robot, rake>(config_a, base_a);
    auto state_b = vamp::collision::make_multi_robot_state<Robot, rake>(config_b, base_b);

    const bool collision_free = vamp::collision::fkcc_multi_all<rake>(environment, state_a, state_b);

    std::cout << (collision_free ? "Collision free" : "In collision") << std::endl;

    return collision_free ? 0 : 1;
}
