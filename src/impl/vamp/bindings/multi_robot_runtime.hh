#pragma once

#include <Eigen/Geometry>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/multi_robot.hh>
#include <vamp/collision/sphere_sphere.hh>
#include <vamp/robots/link_mapping.hh>

namespace vamp::binding::multi_robot
{
    namespace vc = vamp::collision;

    struct RegistryEntry;

    struct PreparedState
    {
        std::vector<vc::Sphere<float>> spheres;
        std::vector<vc::Sphere<float>> attachments;
        const RegistryEntry *registry = nullptr;
    };

    using PrepareFn = bool (*)(
        const std::vector<float> &configuration,
        const Eigen::Isometry3f &base_transform,
        const vc::Attachment<float> *attachment,
        const vc::Environment<float> *environment,
        const vc::MultiRobotCollisionFilter::RobotFilter *filter,
        PreparedState &out_state);

    using LinkLookupFn = bool (*)(std::string_view link_name, std::vector<std::size_t> &out_spheres);

    struct RegistryEntry
    {
        std::string name;
        PrepareFn fn;
        std::size_t dimension;
        std::size_t sphere_count;
        LinkLookupFn link_lookup;
    };

    inline auto registry() -> std::vector<RegistryEntry> &
    {
        static std::vector<RegistryEntry> entries;
        return entries;
    }

    inline auto register_preparer(
        std::string name,
        std::size_t dimension,
        std::size_t sphere_count,
        PrepareFn fn,
        LinkLookupFn link_lookup) -> void
    {
        auto &entries = registry();
        for (auto &entry : entries)
        {
            if (entry.name == name)
            {
                entry.fn = fn;
                entry.dimension = dimension;
                entry.sphere_count = sphere_count;
                entry.link_lookup = link_lookup;
                return;
            }
        }

        entries.push_back(RegistryEntry{std::move(name), fn, dimension, sphere_count, link_lookup});
    }

    inline auto lookup(std::string_view name) -> const RegistryEntry *
    {
        for (auto &entry : registry())
        {
            if (entry.name == name)
            {
                return &entry;
            }
        }

        return nullptr;
    }

    inline constexpr std::size_t vector_width = vamp::FloatVectorWidth;

    template <typename Robot>
    inline auto prepare_state(
        const std::vector<float> &configuration,
        const Eigen::Isometry3f &base_transform,
        const vc::Attachment<float> *attachment,
        const vc::Environment<float> *environment,
        const vc::MultiRobotCollisionFilter::RobotFilter *filter,
        PreparedState &out_state) -> bool
    {
        using Block = typename Robot::template ConfigurationBlock<vector_width>;
        using Row = typename Block::RowT;

        Block block;
        for (std::size_t i = 0; i < Robot::dimension; ++i)
        {
            block[i] = Row::fill(configuration[i]);
        }

        vc::Environment<vamp::FloatVector<vector_width>> robot_environment;
        if (environment != nullptr)
        {
            robot_environment = vc::Environment<vamp::FloatVector<vector_width>>(*environment);
        }

        if (attachment != nullptr)
        {
            robot_environment.attachments.emplace(*attachment);
        }
        else
        {
            robot_environment.attachments.reset();
        }

        bool valid = (attachment != nullptr) ?
                         Robot::template fkcc_attach<vector_width>(robot_environment, block) :
                         Robot::template fkcc<vector_width>(robot_environment, block);

        if (not valid)
        {
            if (filter == nullptr || not vc::detail::collisions_allowed<Robot, vector_width>(
                                             *filter,
                                             robot_environment,
                                             block,
                                             attachment != nullptr))
            {
                return false;
            }

            valid = true;
        }

        typename Robot::template Spheres<vector_width> spheres;
        Robot::template sphere_fk<vector_width>(block, spheres);

        const auto rotation = base_transform.linear();
        const auto translation = base_transform.translation();

        out_state.spheres.clear();
        out_state.spheres.reserve(Robot::n_spheres);
        for (std::size_t i = 0; i < Robot::n_spheres; ++i)
        {
            Eigen::Vector3f position(
                spheres.x[{i, 0}], spheres.y[{i, 0}], spheres.z[{i, 0}]);
            position = rotation * position + translation;
            out_state.spheres.emplace_back(position.x(), position.y(), position.z(), spheres.r[{i, 0}]);
        }

        out_state.attachments.clear();
        if (robot_environment.attachments)
        {
            const auto &posed = robot_environment.attachments->posed_spheres;
            out_state.attachments.reserve(posed.size());
            for (const auto &sphere : posed)
            {
                Eigen::Vector3f position(sphere.x[{0, 0}], sphere.y[{0, 0}], sphere.z[{0, 0}]);
                position = rotation * position + translation;
                out_state.attachments.emplace_back(position.x(), position.y(), position.z(), sphere.r[{0, 0}]);
            }
        }

        return true;
    }

    template <typename Robot>
    inline auto register_robot() -> void
    {
        auto lookup = [](std::string_view link, std::vector<std::size_t> &out) -> bool
        {
            return robots::collect_link_spheres<Robot>(link, out);
        };

        LinkLookupFn link_lookup = nullptr;
        if constexpr (robots::LinkMapping<Robot>::available)
        {
            link_lookup = +lookup;
        }

        register_preparer(Robot::name, Robot::dimension, Robot::n_spheres, &prepare_state<Robot>, link_lookup);
    }
}  // namespace vamp::binding::multi_robot
