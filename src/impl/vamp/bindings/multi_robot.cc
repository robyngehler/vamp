#include <vamp_python_init.hh>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <vamp/bindings/multi_robot_runtime.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/sphere_sphere.hh>

namespace nb = nanobind;
namespace vc = vamp::collision;
namespace vr = vamp::binding::multi_robot;

namespace
{
    struct RawState
    {
        std::string robot;
        std::vector<float> configuration;
        Eigen::Isometry3f transform{Eigen::Isometry3f::Identity()};
        const vc::Attachment<float> *attachment = nullptr;
        nb::object attachment_owner;
    };

    inline auto to_float_vector(nb::handle value) -> std::vector<float>
    {
        nb::sequence seq = nb::cast<nb::sequence>(value);
        std::size_t count = nb::len(seq);

        std::vector<float> result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            result.emplace_back(nb::cast<float>(seq[i]));
        }

        return result;
    }

    inline auto to_transform(nb::handle value) -> Eigen::Isometry3f
    {
        Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();

        if (not value || value.is_none())
        {
            return transform;
        }

        try
        {
            Eigen::Matrix4f matrix = nb::cast<Eigen::Matrix4f>(value);
            transform.matrix() = matrix;
            return transform;
        }
        catch (const nb::cast_error &)
        {
            // Fall through to handle generic sequences below.
        }

        nb::sequence seq = nb::cast<nb::sequence>(value);
        const std::size_t outer = nb::len(seq);

        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        if (outer == 16)
        {
            for (std::size_t i = 0; i < 16; ++i)
            {
                matrix(static_cast<Eigen::Index>(i / 4), static_cast<Eigen::Index>(i % 4)) =
                    nb::cast<float>(seq[i]);
            }
        }
        else if (outer == 4)
        {
            for (std::size_t row = 0; row < 4; ++row)
            {
                nb::sequence row_seq = nb::cast<nb::sequence>(seq[row]);
                if (nb::len(row_seq) != 4)
                {
                    throw nb::value_error("Transform rows must contain exactly four elements.");
                }

                for (std::size_t col = 0; col < 4; ++col)
                {
                    matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                        nb::cast<float>(row_seq[col]);
                }
            }
        }
        else
        {
            throw nb::value_error("Transform must be a 4x4 array-like object.");
        }

        transform.matrix() = matrix;
        return transform;
    }

    inline auto parse_state(nb::handle value) -> RawState
    {
        if (not nb::isinstance<nb::dict>(value))
        {
            throw nb::type_error("Each state must be a mapping with keys 'robot' and 'configuration'.");
        }

        nb::dict mapping = nb::cast<nb::dict>(value);

        if (not mapping.contains("robot"))
        {
            throw nb::value_error("State is missing required key 'robot'.");
        }

        if (not mapping.contains("configuration"))
        {
            throw nb::value_error("State is missing required key 'configuration'.");
        }

        RawState state;
        state.robot = nb::cast<std::string>(mapping["robot"]);
        state.configuration = to_float_vector(mapping["configuration"]);

        nb::handle transform_handle;
        if (mapping.contains("transform"))
        {
            transform_handle = mapping["transform"];
        }
        else if (mapping.contains("base_transform"))
        {
            transform_handle = mapping["base_transform"];
        }

        if (transform_handle and not transform_handle.is_none())
        {
            state.transform = to_transform(transform_handle);
        }

        if (mapping.contains("attachment"))
        {
            nb::object attachment_obj = nb::cast<nb::object>(mapping["attachment"]);
            if (not attachment_obj.is_none())
            {
                state.attachment_owner = attachment_obj;
                state.attachment = nb::cast<const vc::Attachment<float> *>(attachment_obj);
            }
        }

        return state;
    }

    inline auto parse_states(const nb::sequence &states_seq) -> std::vector<RawState>
    {
        const std::size_t count = nb::len(states_seq);
        std::vector<RawState> states;
        states.reserve(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            states.emplace_back(parse_state(states_seq[i]));
        }

        return states;
    }

    inline auto spheres_collide(
        const std::vector<vc::Sphere<float>> &a,
        const std::vector<vc::Sphere<float>> &b) -> bool
    {
        for (const auto &sa : a)
        {
            for (const auto &sb : b)
            {
                if (vc::sphere_sphere_sql2(sa, sb) <= 0.F)
                {
                    return true;
                }
            }
        }

        return false;
    }

    inline auto attachments_vs_spheres(
        const std::vector<vc::Sphere<float>> &attachments,
        const std::vector<vc::Sphere<float>> &spheres) -> bool
    {
        if (attachments.empty())
        {
            return false;
        }

        return spheres_collide(attachments, spheres);
    }

    inline auto attachments_vs_attachments(
        const std::vector<vc::Sphere<float>> &a,
        const std::vector<vc::Sphere<float>> &b) -> bool
    {
        if (a.empty() || b.empty())
        {
            return false;
        }

        return spheres_collide(a, b);
    }

    inline auto collision_free(const std::vector<vr::PreparedState> &states) -> bool
    {
        const std::size_t count = states.size();

        for (std::size_t i = 0; i < count; ++i)
        {
            for (std::size_t j = i + 1; j < count; ++j)
            {
                if (spheres_collide(states[i].spheres, states[j].spheres))
                {
                    return false;
                }
            }
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            if (states[i].attachments.empty())
            {
                continue;
            }

            for (std::size_t j = 0; j < count; ++j)
            {
                if (i == j)
                {
                    continue;
                }

                if (attachments_vs_spheres(states[i].attachments, states[j].spheres))
                {
                    return false;
                }
            }
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            if (states[i].attachments.empty())
            {
                continue;
            }

            for (std::size_t j = i + 1; j < count; ++j)
            {
                if (attachments_vs_attachments(states[i].attachments, states[j].attachments))
                {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename Callback>
    inline auto prepare_states(
        const std::vector<RawState> &raw_states,
        const vc::Environment<float> *environment,
        Callback &&on_invalid) -> bool
    {
        std::vector<vr::PreparedState> prepared(raw_states.size());

        for (std::size_t i = 0; i < raw_states.size(); ++i)
        {
            const auto &raw = raw_states[i];
            const auto *entry = vr::lookup(raw.robot);
            if (entry == nullptr)
            {
                const auto message = std::string("Unknown robot '") + raw.robot + "'.";
                throw nb::value_error(message.c_str());
            }

            if (raw.configuration.size() != entry->dimension)
            {
                std::ostringstream oss;
                oss << "Configuration for robot '" << raw.robot << "' must contain " << entry->dimension
                    << " values (received " << raw.configuration.size() << ").";
                const auto message = oss.str();
                throw nb::value_error(message.c_str());
            }

            if (not entry->fn(raw.configuration, raw.transform, raw.attachment, environment, prepared[i]))
            {
                return on_invalid();
            }
        }

        return collision_free(prepared);
    }
}  // namespace

void vamp::binding::init_multi_robot(nanobind::module_ &pymodule)
{
    using namespace nb::literals;

    pymodule.def(
        "fkcc_multi_all",
        [](const nb::sequence &states_seq, const vc::Environment<float> &environment)
        {
            auto states = parse_states(states_seq);
            if (states.empty())
            {
                throw nb::value_error("fkcc_multi_all requires at least one robot state.");
            }

            return prepare_states(
                states,
                &environment,
                []() { return false; });
        },
        "states"_a,
        "environment"_a,
        "Checks whether multiple robots are collision-free against each other and the shared environment.");

    pymodule.def(
        "fkcc_multi_self",
        [](const nb::sequence &states_seq)
        {
            auto states = parse_states(states_seq);
            if (states.empty())
            {
                throw nb::value_error("fkcc_multi_self requires at least one robot state.");
            }

            return prepare_states(
                states,
                nullptr,
                []() { return false; });
        },
        "states"_a,
        "Checks whether multiple robots are collision-free against each other, ignoring the environment.");
}
