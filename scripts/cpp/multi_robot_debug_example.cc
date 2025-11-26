#include <array>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <vamp/collision/multi_robot.hh>
#include <vamp/robots/gp4.hh>

namespace vc = vamp::collision;

using Robot = vamp::robots::GP4;
constexpr std::size_t rake = vamp::FloatVectorWidth;
using Block = Robot::template ConfigurationBlock<rake>;
using Row = Block::RowT;

inline auto pack_configuration(const std::array<float, Robot::dimension> &values) -> Block
{
    Block block = Block::zero_vector();
    for (std::size_t i = 0; i < Robot::dimension; ++i)
    {
        block[i] = Row::fill(values[i]);
    }
    return block;
}

inline auto make_environment() -> vc::Environment<vamp::FloatVector<rake>>
{
    vc::Environment<float> environment;

    environment.spheres.emplace_back(0.0F, 0.0F, 0.2F, 0.25F);
    environment.spheres.back().name = "center_sphere";

    environment.spheres.emplace_back(-0.35F, 0.2F, 0.25F, 0.12F);
    environment.spheres.back().name = "offset_sphere";

    environment.sort();
    return vc::Environment<vamp::FloatVector<rake>>(environment);
}

inline auto print_body(const vc::DebugBody &body, const std::vector<std::string> &labels) -> void
{
    const std::string &label = (body.robot_index < labels.size()) ? labels[body.robot_index] : std::string("robot");
    std::cout << label << " (index " << body.robot_index << "): ";

    if (body.is_attachment)
    {
        std::cout << "attachment[" << body.attachment_index << "]";
    }
    else
    {
        std::cout << "link[" << body.link_index << "] (" << body.name << ")";
    }

    std::cout << " sphere_idx=" << body.sphere_index;
}

inline auto print_result(const vc::DebugFkccMultiAllResult &result, const std::vector<std::string> &labels) -> void
{
    if (result.link_link.empty() && result.link_object.empty())
    {
        std::cout << "  collision-free\n";
        return;
    }

    for (const auto &collision : result.link_link)
    {
        std::cout << "  link-link: ";
        print_body(collision.first, labels);
        std::cout << " <-> ";
        print_body(collision.second, labels);
        std::cout << "\n";
    }

    for (const auto &collision : result.link_object)
    {
        std::cout << "  link-object: ";
        print_body(collision.link, labels);
        std::cout << " vs object '" << collision.object << "'\n";
    }
}

int main()
{
    std::vector<std::string> robot_labels = {"arm_a", "arm_b"};

    auto environment = make_environment();

    Eigen::Isometry3f base_a = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f base_b = Eigen::Isometry3f::Identity();
    base_b.translate(Eigen::Vector3f(0.35F, -0.05F, 0.0F));

    vc::Attachment<float> attachment(Eigen::Isometry3f::Identity());
    attachment.spheres.emplace_back(0.0F, 0.0F, 0.12F, 0.06F);

    // Hardcoded pose pair: intentionally place the arms close together and near the center sphere.
    std::array<float, Robot::dimension> config_a = {0.0F, -1.4F, 0.2F, -1.5F, 0.2F, 0.0F};
    std::array<float, Robot::dimension> config_b = {0.0F, -1.3F, -0.1F, -1.6F, -0.1F, 0.0F};

    auto state_a = vc::make_multi_robot_state<Robot, rake>(pack_configuration(config_a), base_a, &attachment);
    auto state_b = vc::make_multi_robot_state<Robot, rake>(pack_configuration(config_b), base_b);

    std::cout << "Deterministic pair:\n";
    const auto deterministic = vc::debug_fkcc_multi_all<rake>(environment, state_a, state_b);
    print_result(deterministic, robot_labels);

    // Randomized samples around nominal to show reporting in diverse poses.
    std::mt19937 rng(7U);
    std::uniform_real_distribution<float> joint_noise(-0.25F, 0.25F);
    const std::array<float, Robot::dimension> nominal = {0.0F, -1.3F, 0.0F, -1.6F, 0.0F, 0.0F};

    std::cout << "\nRandom samples:\n";
    for (int sample = 0; sample < 5; ++sample)
    {
        std::array<float, Robot::dimension> sample_a = nominal;
        std::array<float, Robot::dimension> sample_b = nominal;

        for (std::size_t joint = 0; joint < Robot::dimension; ++joint)
        {
            sample_a[joint] += joint_noise(rng);
            sample_b[joint] += joint_noise(rng);
        }

        auto random_state_a =
            vc::make_multi_robot_state<Robot, rake>(pack_configuration(sample_a), base_a, &attachment);
        auto random_state_b = vc::make_multi_robot_state<Robot, rake>(pack_configuration(sample_b), base_b);

        std::cout << "Sample " << sample << ":\n";
        const auto result = vc::debug_fkcc_multi_all<rake>(environment, random_state_a, random_state_b);
        print_result(result, robot_labels);
    }

    return 0;
}
