#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>
#include <vamp/collision/sphere_sphere.hh>

namespace vamp::collision
{
    struct MultiRobotCollisionFilter
    {
        struct RobotFilter
        {
            std::vector<std::unordered_set<std::string>> per_sphere_objects;
            std::unordered_set<std::string> attachment_objects;

            inline auto ensure_sphere_capacity(std::size_t count) -> void
            {
                if (per_sphere_objects.size() < count)
                {
                    per_sphere_objects.resize(count);
                }
            }

            inline auto active() const noexcept -> bool
            {
                if (not attachment_objects.empty())
                {
                    return true;
                }

                for (const auto &entries : per_sphere_objects)
                {
                    if (not entries.empty())
                    {
                        return true;
                    }
                }

                return false;
            }

            inline auto allow_sphere(std::size_t sphere_index, std::string_view object) -> void
            {
                ensure_sphere_capacity(sphere_index + 1);
                per_sphere_objects[sphere_index].emplace(object);
            }

            inline auto allow_attachment(std::string_view object) -> void
            {
                attachment_objects.emplace(object);
            }

            inline auto allows_sphere(std::size_t sphere_index, std::string_view object) const -> bool
            {
                if (sphere_index >= per_sphere_objects.size())
                {
                    return false;
                }

                const auto &entries = per_sphere_objects[sphere_index];
                if (entries.empty())
                {
                    return false;
                }

                if (entries.find(std::string(object)) != entries.end())
                {
                    return true;
                }

                return entries.find(std::string("*")) != entries.end();
            }

            inline auto allows_attachment(std::string_view object) const -> bool
            {
                if (attachment_objects.empty())
                {
                    return false;
                }

                if (attachment_objects.find(std::string(object)) != attachment_objects.end())
                {
                    return true;
                }

                return attachment_objects.find(std::string("*")) != attachment_objects.end();
            }
        };

        inline auto empty() const noexcept -> bool
        {
            return std::none_of(per_robot.begin(), per_robot.end(), [](const RobotFilter &filter) {
                return filter.active();
            });
        }

        inline auto ensure_robot(std::size_t index, std::size_t sphere_count) -> RobotFilter &
        {
            if (index >= per_robot.size())
            {
                per_robot.resize(index + 1U);
            }

            auto &filter = per_robot[index];
            filter.ensure_sphere_capacity(sphere_count);
            return filter;
        }

        inline auto robot(std::size_t index) noexcept -> RobotFilter *
        {
            if (index >= per_robot.size())
            {
                return nullptr;
            }

            return &per_robot[index];
        }

        inline auto robot(std::size_t index) const noexcept -> const RobotFilter *
        {
            if (index >= per_robot.size())
            {
                return nullptr;
            }

            return &per_robot[index];
        }

        std::vector<RobotFilter> per_robot;
    };

    template <typename RobotT, std::size_t rake>
    struct MultiRobotState
    {
        using RobotType = RobotT;
        using ConfigurationBlock = typename RobotType::template ConfigurationBlock<rake>;

        ConfigurationBlock configuration;
        Eigen::Isometry3f base_transform{Eigen::Isometry3f::Identity()};
        const collision::Attachment<float> *attachment = nullptr;
    };

    template <typename RobotT, std::size_t rake>
    inline auto make_multi_robot_state(
        const typename RobotT::template ConfigurationBlock<rake> &configuration,
        const Eigen::Isometry3f &base_transform = Eigen::Isometry3f::Identity(),
        const collision::Attachment<float> *attachment = nullptr) noexcept
        -> MultiRobotState<RobotT, rake>
    {
        return {configuration, base_transform, attachment};
    }

    namespace detail
    {
        template <typename SpheresT>
        inline auto apply_transform(SpheresT &spheres, const Eigen::Isometry3f &tf) noexcept -> void
        {
            const auto rotation = tf.linear();
            const auto translation = tf.translation();

            const auto rx0 = rotation(0, 0);
            const auto rx1 = rotation(0, 1);
            const auto rx2 = rotation(0, 2);
            const auto ry0 = rotation(1, 0);
            const auto ry1 = rotation(1, 1);
            const auto ry2 = rotation(1, 2);
            const auto rz0 = rotation(2, 0);
            const auto rz1 = rotation(2, 1);
            const auto rz2 = rotation(2, 2);

            auto &x = spheres.x;
            auto &y = spheres.y;
            auto &z = spheres.z;

            auto transformed_x = x * rx0 + y * rx1 + z * rx2;
            auto transformed_y = x * ry0 + y * ry1 + z * ry2;
            auto transformed_z = x * rz0 + y * rz1 + z * rz2;

            transformed_x = transformed_x + translation.x();
            transformed_y = transformed_y + translation.y();
            transformed_z = transformed_z + translation.z();

            x = transformed_x;
            y = transformed_y;
            z = transformed_z;
        }

        template <std::size_t rake>
        inline auto apply_transform(
            std::vector<collision::Sphere<FloatVector<rake>>> &spheres,
            const Eigen::Isometry3f &tf) noexcept -> void
        {
            const auto rotation = tf.linear();
            const auto translation = tf.translation();

            const auto rx0 = rotation(0, 0);
            const auto rx1 = rotation(0, 1);
            const auto rx2 = rotation(0, 2);
            const auto ry0 = rotation(1, 0);
            const auto ry1 = rotation(1, 1);
            const auto ry2 = rotation(1, 2);
            const auto rz0 = rotation(2, 0);
            const auto rz1 = rotation(2, 1);
            const auto rz2 = rotation(2, 2);

            for (auto &sphere : spheres)
            {
                const auto x = sphere.x;
                const auto y = sphere.y;
                const auto z = sphere.z;

                sphere.x = x * rx0 + y * rx1 + z * rx2 + translation.x();
                sphere.y = x * ry0 + y * ry1 + z * ry2 + translation.y();
                sphere.z = x * rz0 + y * rz1 + z * rz2 + translation.z();
            }
        }

        inline auto is_identity_transform(const Eigen::Isometry3f &tf) noexcept -> bool
        {
            return tf.linear().isIdentity(1e-6F) && tf.translation().isZero(1e-6F);
        }

        inline auto transform_environment_to_robot_frame(
            collision::Environment<float> &environment,
            const Eigen::Isometry3f &base_transform) noexcept -> void
        {
            const Eigen::Matrix3f rotation_inv = base_transform.linear().transpose();
            const Eigen::Vector3f translation = base_transform.translation();

            auto transform_point = [&](float x, float y, float z) noexcept -> Eigen::Vector3f
            {
                Eigen::Vector3f result{x, y, z};
                result -= translation;
                result = rotation_inv * result;
                return result;
            };

            auto transform_direction = [&](float x, float y, float z) noexcept -> Eigen::Vector3f
            {
                return rotation_inv * Eigen::Vector3f{x, y, z};
            };

            for (auto &sphere : environment.spheres)
            {
                const Eigen::Vector3f point = transform_point(sphere.x, sphere.y, sphere.z);
                sphere.x = point.x();
                sphere.y = point.y();
                sphere.z = point.z();
                const float distance = point.norm();
                sphere.min_distance = distance - sphere.r;
            }

            auto transform_segment = [&](auto &segment) noexcept
            {
                const Eigen::Vector3f start(segment.x1, segment.y1, segment.z1);
                const Eigen::Vector3f end = start + Eigen::Vector3f(segment.xv, segment.yv, segment.zv);

                const Eigen::Vector3f start_local = transform_point(start.x(), start.y(), start.z());
                const Eigen::Vector3f end_local = transform_point(end.x(), end.y(), end.z());
                const Eigen::Vector3f delta = end_local - start_local;

                segment.x1 = start_local.x();
                segment.y1 = start_local.y();
                segment.z1 = start_local.z();
                segment.xv = delta.x();
                segment.yv = delta.y();
                segment.zv = delta.z();
            };

            for (auto &capsule : environment.capsules)
            {
                transform_segment(capsule);
                capsule.min_distance = capsule.compute_min_distance();
            }
            for (auto &capsule : environment.z_aligned_capsules)
            {
                transform_segment(capsule);
                capsule.min_distance = capsule.compute_min_distance();
            }
            for (auto &cylinder : environment.cylinders)
            {
                transform_segment(cylinder);
                cylinder.min_distance = cylinder.compute_min_distance();
            }

            auto transform_cuboid = [&](auto &cuboid) noexcept
            {
                const Eigen::Vector3f center = transform_point(cuboid.x, cuboid.y, cuboid.z);
                cuboid.x = center.x();
                cuboid.y = center.y();
                cuboid.z = center.z();

                const Eigen::Vector3f axis1 = transform_direction(cuboid.axis_1_x, cuboid.axis_1_y, cuboid.axis_1_z);
                cuboid.axis_1_x = axis1.x();
                cuboid.axis_1_y = axis1.y();
                cuboid.axis_1_z = axis1.z();

                const Eigen::Vector3f axis2 = transform_direction(cuboid.axis_2_x, cuboid.axis_2_y, cuboid.axis_2_z);
                cuboid.axis_2_x = axis2.x();
                cuboid.axis_2_y = axis2.y();
                cuboid.axis_2_z = axis2.z();

                const Eigen::Vector3f axis3 = transform_direction(cuboid.axis_3_x, cuboid.axis_3_y, cuboid.axis_3_z);
                cuboid.axis_3_x = axis3.x();
                cuboid.axis_3_y = axis3.y();
                cuboid.axis_3_z = axis3.z();
            };

            for (auto &cuboid : environment.cuboids)
            {
                transform_cuboid(cuboid);
                cuboid.min_distance = cuboid.compute_min_distance();
            }
            for (auto &cuboid : environment.z_aligned_cuboids)
            {
                transform_cuboid(cuboid);
                cuboid.min_distance = cuboid.compute_min_distance();
            }

            environment.sort();
        }

        template <std::size_t rake>
        struct EnvironmentTransformCache
        {
            struct Entry
            {
                Eigen::Isometry3f transform;
                collision::Environment<FloatVector<rake>> environment;
            };

            inline static thread_local std::vector<Entry> storage{};
        };

        template <std::size_t rake>
        inline auto environment_entry_for_transform(
            const collision::Environment<FloatVector<rake>> &source_environment,
            const Eigen::Isometry3f &base_transform)
            -> typename EnvironmentTransformCache<rake>::Entry &
        {
            auto &storage = EnvironmentTransformCache<rake>::storage;
            for (auto &entry : storage)
            {
                if (entry.transform.linear().isApprox(base_transform.linear(), 1e-6F) &&
                    entry.transform.translation().isApprox(base_transform.translation(), 1e-6F))
                {
                    return entry;
                }
            }

            collision::Environment<FloatVector<rake>> transformed =
                is_identity_transform(base_transform) ?
                    collision::Environment<FloatVector<rake>>(source_environment) :
                    [&]
                    {
                        collision::Environment<float> float_environment(source_environment);
                        transform_environment_to_robot_frame(float_environment, base_transform);
                        return collision::Environment<FloatVector<rake>>(float_environment);
                    }();

            storage.push_back({base_transform, std::move(transformed)});
            return storage.back();
        }

        template <typename Robot, std::size_t rake>
        inline auto collisions_allowed(
            const MultiRobotCollisionFilter::RobotFilter &filter,
            const collision::Environment<FloatVector<rake>> &environment,
            const typename Robot::template ConfigurationBlock<rake> &configuration,
            bool attachment_present) noexcept -> bool;

        template <typename RobotStateT, std::size_t rake>
        inline auto process_robot_state(
            const collision::Environment<FloatVector<rake>> &source_environment,
            const RobotStateT &state,
            typename RobotStateT::RobotType::template Spheres<rake> &spheres,
            std::vector<collision::Sphere<FloatVector<rake>>> &attachment_spheres,
            const MultiRobotCollisionFilter::RobotFilter *filter) noexcept -> bool
        {
            using Robot = typename RobotStateT::RobotType;

            auto &entry = detail::environment_entry_for_transform<rake>(
                source_environment, state.base_transform);
            auto &robot_environment = entry.environment;

            auto saved_attachments = robot_environment.attachments;

            if (state.attachment)
            {
                robot_environment.attachments.emplace(*state.attachment);
            }
            else
            {
                robot_environment.attachments.reset();
            }

            bool valid = state.attachment ?
                             Robot::template fkcc_attach<rake>(robot_environment, state.configuration) :
                             Robot::template fkcc<rake>(robot_environment, state.configuration);

            if (not valid)
            {
                if (filter == nullptr)
                {
                    robot_environment.attachments = std::move(saved_attachments);
                    return false;
                }

                if (not collisions_allowed<Robot, rake>(
                        *filter,
                        robot_environment,
                        state.configuration,
                        state.attachment != nullptr))
                {
                    robot_environment.attachments = std::move(saved_attachments);
                    return false;
                }

                valid = true;
            }

            Robot::sphere_fk(state.configuration, spheres);
            apply_transform(spheres, state.base_transform);

            attachment_spheres.clear();
            if (robot_environment.attachments)
            {
                attachment_spheres = robot_environment.attachments->posed_spheres;
                apply_transform<rake>(attachment_spheres, state.base_transform);
            }

            robot_environment.attachments = std::move(saved_attachments);

            return true;
        }

        template <typename RobotStateT, std::size_t rake>
        inline auto process_robot_state_self(
            const RobotStateT &state,
            typename RobotStateT::RobotType::template Spheres<rake> &spheres,
            std::vector<collision::Sphere<FloatVector<rake>>> &attachment_spheres) noexcept -> bool
        {
            using Robot = typename RobotStateT::RobotType;

            collision::Environment<FloatVector<rake>> robot_environment;

            if (state.attachment)
            {
                robot_environment.attachments.emplace(*state.attachment);
            }

            const bool valid = state.attachment ?
                                  Robot::template fkcc_attach<rake>(robot_environment, state.configuration) :
                                  Robot::template fkcc<rake>(robot_environment, state.configuration);

            if (not valid)
            {
                // fkcc_multi_* returns false when any SIMD lane detects a collision.
                return false;
            }

            Robot::sphere_fk(state.configuration, spheres);
            apply_transform(spheres, state.base_transform);

            attachment_spheres.clear();
            if (robot_environment.attachments)
            {
                attachment_spheres = robot_environment.attachments->posed_spheres;
                apply_transform<rake>(attachment_spheres, state.base_transform);
            }

            return true;
        }

        template <typename SpheresA, typename SpheresB>
        inline auto spheres_collide(const SpheresA &a, const SpheresB &b) noexcept -> bool
        {
            constexpr std::size_t count_a = std::remove_reference_t<decltype(a.x)>::num_rows;
            constexpr std::size_t count_b = std::remove_reference_t<decltype(b.x)>::num_rows;

            // Each coordinate component is a FloatVector<rake>, so every arithmetic
            // operation below runs across all SIMD lanes in lockstep. The outer loops
            // are only over the small, fixed set of robot spheres.
            for (std::size_t i = 0; i < count_a; ++i)
            {
                const auto ax = a.x[i];
                const auto ay = a.y[i];
                const auto az = a.z[i];
                const auto ar = a.r[i];

                for (std::size_t j = 0; j < count_b; ++j)
                {
                    const auto bx = b.x[j];
                    const auto by = b.y[j];
                    const auto bz = b.z[j];
                    const auto br = b.r[j];

                    if (not collision::sphere_sphere_sql2(ax, ay, az, ar, bx, by, bz, br).test_zero())
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        template <std::size_t rake>
        inline auto spheres_collide(
            const collision::Sphere<FloatVector<rake>> &a,
            const collision::Sphere<FloatVector<rake>> &b) noexcept -> bool
        {
            return not collision::sphere_sphere_sql2(a, b).test_zero();
        }

        template <typename AttachmentVec, typename SpheresT>
        inline auto attachments_vs_spheres(const AttachmentVec &attachments, const SpheresT &spheres) noexcept -> bool
        {
            if (attachments.empty())
            {
                return false;
            }

            constexpr std::size_t count = std::remove_reference_t<decltype(spheres.x)>::num_rows;

            for (const auto &attachment : attachments)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (not collision::sphere_sphere_sql2(
                                attachment,
                                spheres.x[i],
                                spheres.y[i],
                                spheres.z[i],
                                spheres.r[i])
                                .test_zero())
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        template <typename AttachmentVecA, typename AttachmentVecB>
        inline auto attachments_vs_attachments(
            const AttachmentVecA &a,
            const AttachmentVecB &b) noexcept -> bool
        {
            if (a.empty() || b.empty())
            {
                return false;
            }

            for (const auto &sphere_a : a)
            {
                for (const auto &sphere_b : b)
                {
                    if (not collision::sphere_sphere_sql2(sphere_a, sphere_b).test_zero())
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        template <std::size_t I, std::size_t J, typename Tuple>
        inline auto spheres_cross_pair(const Tuple &spheres) noexcept -> bool
        {
            if constexpr (I >= std::tuple_size_v<Tuple>)
            {
                return false;
            }
            else if constexpr (J >= std::tuple_size_v<Tuple>)
            {
                return false;
            }
            else
            {
                // Compare the I-th and J-th robots; the FloatVector math inside
                // the check stays vectorized across rake lanes.
                if (spheres_collide(std::get<I>(spheres), std::get<J>(spheres)))
                {
                    return true;
                }

                return spheres_cross_pair<I, J + 1>(spheres);
            }
        }

        template <std::size_t I, typename Tuple>
        inline auto spheres_cross(const Tuple &spheres) noexcept -> bool
        {
            if constexpr (I >= std::tuple_size_v<Tuple>)
            {
                return false;
            }
            else
            {
                // Start at robot I and walk forward so that every unique pair is
                // tested exactly once; compile-time recursion keeps the loops tight.
                if (spheres_cross_pair<I, I + 1>(spheres))
                {
                    return true;
                }

                return spheres_cross<I + 1>(spheres);
            }
        }

        template <std::size_t I, std::size_t J, typename AttachTuple, typename SphereTuple>
        inline auto attachments_vs_spheres_pair(
            const AttachTuple &attachments,
            const SphereTuple &spheres) noexcept -> bool
        {
            if constexpr (J >= std::tuple_size_v<SphereTuple>)
            {
                return false;
            }
            else
            {
                if constexpr (J != I)
                {
                    if (attachments_vs_spheres(std::get<I>(attachments), std::get<J>(spheres)))
                    {
                        return true;
                    }
                }

                return attachments_vs_spheres_pair<I, J + 1>(attachments, spheres);
            }
        }

        template <std::size_t I, typename AttachTuple, typename SphereTuple>
        inline auto attachments_vs_spheres_all(
            const AttachTuple &attachments,
            const SphereTuple &spheres) noexcept -> bool
        {
            if constexpr (I >= std::tuple_size_v<AttachTuple>)
            {
                return false;
            }
            else
            {
                if (attachments_vs_spheres_pair<I, 0>(attachments, spheres))
                {
                    return true;
                }

                return attachments_vs_spheres_all<I + 1>(attachments, spheres);
            }
        }

        template <std::size_t I, std::size_t J, typename AttachTuple>
        inline auto attachments_vs_attachments_pair(const AttachTuple &attachments) noexcept -> bool
        {
            if constexpr (J >= std::tuple_size_v<AttachTuple>)
            {
                return false;
            }
            else
            {
                if constexpr (J > I)
                {
                    if (attachments_vs_attachments(std::get<I>(attachments), std::get<J>(attachments)))
                    {
                        return true;
                    }
                }

                return attachments_vs_attachments_pair<I, J + 1>(attachments);
            }
        }

        template <std::size_t I, typename AttachTuple>
        inline auto attachments_vs_attachments_all(const AttachTuple &attachments) noexcept -> bool
        {
            if constexpr (I >= std::tuple_size_v<AttachTuple>)
            {
                return false;
            }
            else
            {
                if (attachments_vs_attachments_pair<I, 0>(attachments))
                {
                    return true;
                }

                return attachments_vs_attachments_all<I + 1>(attachments);
            }
        }

        template <typename Tuple>
        inline auto attachments_present(const Tuple &attachments) noexcept -> bool
        {
            bool has_attachment = false;
            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                ((has_attachment = has_attachment || not std::get<I>(attachments).empty()), ...);
            }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});

            return has_attachment;
        }

        inline auto set_contains(const std::unordered_set<std::string> &set, std::string_view value) -> bool
        {
            if (set.empty())
            {
                return false;
            }

            if (set.find(std::string(value)) != set.end())
            {
                return true;
            }

            return set.find(std::string("*")) != set.end();
        }

        template <typename Robot, std::size_t rake>
        inline auto collisions_allowed(
            const MultiRobotCollisionFilter::RobotFilter &filter,
            const collision::Environment<FloatVector<rake>> &environment,
            const typename Robot::template ConfigurationBlock<rake> &configuration,
            bool attachment_present) noexcept -> bool
        {
            if (not filter.active())
            {
                return false;
            }

            const auto debug = Robot::template fkcc_debug<rake>(environment, configuration);

            bool has_collision = false;

            for (std::size_t sphere_index = 0; sphere_index < debug.first.size(); ++sphere_index)
            {
                const auto &objects = debug.first[sphere_index];
                if (objects.empty())
                {
                    continue;
                }

                has_collision = true;
                for (const auto &object : objects)
                {
                    if (filter.per_sphere_objects.size() <= sphere_index)
                    {
                        return false;
                    }

                    const auto &allowed = filter.per_sphere_objects[sphere_index];
                    if (not set_contains(allowed, object))
                    {
                        return false;
                    }
                }
            }

            if (not debug.second.empty())
            {
                return false;
            }

            if (attachment_present && environment.attachments)
            {
                for (const auto &sphere : environment.attachments->posed_spheres)
                {
                    const auto collisions = sphere_environment_get_collisions(
                        environment,
                        sphere.x,
                        sphere.y,
                        sphere.z,
                        sphere.r);

                    if (collisions.empty())
                    {
                        continue;
                    }

                    has_collision = true;
                    for (const auto &object : collisions)
                    {
                        if (not set_contains(filter.attachment_objects, object))
                        {
                            return false;
                        }
                    }
                }
            }

            return has_collision;
        }

        template <std::size_t rake, typename... States>
        inline auto fkcc_multi_all_attach_impl(
            const collision::Environment<FloatVector<rake>> &environment,
            const MultiRobotCollisionFilter *filter,
            const States &... states) noexcept -> bool
        {
            static_assert(sizeof...(States) > 0, "fkcc_multi_all_attach requires at least one robot state");

            auto spheres = std::tuple<typename States::RobotType::template Spheres<rake>...>{};
            auto attachments = std::tuple<
                std::conditional_t<true, std::vector<collision::Sphere<FloatVector<rake>>>, States>...>{};

            bool valid = true;

            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                (..., ([&]
                       {
                           using StateT = std::tuple_element_t<I, std::tuple<States...>>;
                           const auto *robot_filter =
                               (filter != nullptr) ? filter->robot(I) : nullptr;

                           if (not process_robot_state<StateT, rake>(
                                   environment,
                                   std::get<I>(std::forward_as_tuple(states...)),
                                   std::get<I>(spheres),
                                   std::get<I>(attachments),
                                   robot_filter))
                           {
                               valid = false;
                           }
                       }()));
            }(std::index_sequence_for<States...>{});

            if (not valid)
            {
                // fkcc_multi_self returns false when any SIMD lane collides.
                return false;
            }

            // Kick off the recursive robot/robot sphere sweep starting at tuple index 0.
            // Same pairwise robot sweep as above, but on the self-checked robot set.
            if (spheres_cross<0>(spheres))
            {
                return false;
            }

            if (attachments_present(attachments))
            {
                if (attachments_vs_spheres_all<0>(attachments, spheres))
                {
                    return false;
                }

                if (attachments_vs_attachments_all<0>(attachments))
                {
                    return false;
                }
            }

            return true;
        }

        template <std::size_t rake, typename... States>
        inline auto fkcc_multi_self_attach_impl(const States &... states) noexcept -> bool
        {
            static_assert(sizeof...(States) > 0, "fkcc_multi_self_attach requires at least one robot state");

            auto spheres = std::tuple<typename States::RobotType::template Spheres<rake>...>{};
            auto attachments = std::tuple<
                std::conditional_t<true, std::vector<collision::Sphere<FloatVector<rake>>>, States>...>{};

            bool valid = true;

            [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                (..., ([&]
                       {
                           using StateT = std::tuple_element_t<I, std::tuple<States...>>;
                           if (not detail::process_robot_state_self<StateT, rake>(
                                   std::get<I>(std::forward_as_tuple(states...)),
                                   std::get<I>(spheres),
                                   std::get<I>(attachments)))
                           {
                               valid = false;
                           }
                       }()));
            }(std::index_sequence_for<States...>{});

            if (not valid)
            {
                return false;
            }

            // Same pairwise robot sweep as above, but on the self-checked robot set.
            if (spheres_cross<0>(spheres))
            {
                return false;
            }

            if (attachments_present(attachments))
            {
                if (attachments_vs_spheres_all<0>(attachments, spheres))
                {
                    return false;
                }

                if (attachments_vs_attachments_all<0>(attachments))
                {
                    return false;
                }
            }

            return true;
        }
    }  // namespace detail

    template <std::size_t rake, typename... States>
    inline auto fkcc_multi_all_attach(
        const collision::Environment<FloatVector<rake>> &environment,
        const States &... states) noexcept -> bool
    {
        return detail::fkcc_multi_all_attach_impl<rake>(environment, nullptr, states...);
    }

    template <std::size_t rake, typename... States>
    inline auto fkcc_multi_all_attach(
        const collision::Environment<FloatVector<rake>> &environment,
        const MultiRobotCollisionFilter &filter,
        const States &... states) noexcept -> bool
    {
        return detail::fkcc_multi_all_attach_impl<rake>(environment, &filter, states...);
    }

    template <std::size_t rake, typename... States>
    inline auto fkcc_multi_all(
        const collision::Environment<FloatVector<rake>> &environment,
        const States &... states) noexcept -> bool
    {
        return detail::fkcc_multi_all_attach_impl<rake>(environment, nullptr, states...);
    }

    template <std::size_t rake, typename... States>
    inline auto fkcc_multi_all(
        const collision::Environment<FloatVector<rake>> &environment,
        const MultiRobotCollisionFilter &filter,
        const States &... states) noexcept -> bool
    {
        return detail::fkcc_multi_all_attach_impl<rake>(environment, &filter, states...);
    }

    template <std::size_t rake, typename... States>
    inline auto fkcc_multi_self_attach(const States &... states) noexcept -> bool
    {
        return detail::fkcc_multi_self_attach_impl<rake>(states...);
    }

    template <std::size_t rake, typename... States>
    inline auto fkcc_multi_self(const States &... states) noexcept -> bool
    {
        return detail::fkcc_multi_self_attach_impl<rake>(states...);
    }
}  // namespace vamp::collision
