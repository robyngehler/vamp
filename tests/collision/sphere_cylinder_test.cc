// Boundary tests for the exact sphere-cylinder test and for cylinders in the environment check.
// Built as `vamp_sphere_cylinder_test` with -DVAMP_BUILD_TESTS=ON and run by ctest.

#include <array>
#include <cmath>
#include <cstdio>
#include <random>

#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/collision/sphere_cylinder.hh>
#include <vamp/collision/validity.hh>
#include <vamp/vector.hh>

namespace
{
    using V = vamp::FloatVector<>;
    using vamp::collision::Cylinder;

    int failures = 0;

    void check(bool ok, const char *what)
    {
        if (not ok)
        {
            ++failures;
            std::printf("FAIL %s\n", what);
        }
    }

    struct Frame
    {
        std::array<double, 9> R;  // row-major rotation
        std::array<double, 3> t;

        auto apply(std::array<double, 3> p) const -> std::array<double, 3>
        {
            return {
                R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0],
                R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1],
                R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]};
        }
    };

    auto rotation(double ax, double ay, double az, double angle, std::array<double, 3> t) -> Frame
    {
        const double n = std::sqrt(ax * ax + ay * ay + az * az);
        ax /= n, ay /= n, az /= n;
        const double c = std::cos(angle), s = std::sin(angle), C = 1.0 - c;
        return {{c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s,
                 ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s,
                 az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C},
                t};
    }

    // Cylinder of radius r along the local z axis from z = 0 to z = length, placed by frame f.
    auto make(const Frame &f, double r, double length) -> Cylinder<float>
    {
        const auto a = f.apply({0, 0, 0});
        const auto b = f.apply({0, 0, length});
        return vamp::collision::factory::cylinder::endpoints::flat(
            float(a[0]), float(a[1]), float(a[2]), float(b[0]), float(b[1]), float(b[2]), float(r));
    }

    // Signed sphere-cylinder value in every lane for one probe, read back from lane 0.
    auto value(const Cylinder<float> &c, std::array<double, 3> p, double r) -> float
    {
        const Cylinder<V> cv(c);
        const auto out = vamp::collision::sphere_cylinder(cv, V(float(p[0])), V(float(p[1])),
                                                          V(float(p[2])), V(float(r)));
        return out.to_array()[0];
    }

    // Exact distance from a point to the solid local cylinder, in double.
    auto reference_distance(std::array<double, 3> local, double r, double length) -> double
    {
        const double radial = std::max(std::hypot(local[0], local[1]) - r, 0.0);
        const double axial = std::max(std::max(-local[2], local[2] - length), 0.0);
        return std::hypot(radial, axial);
    }

    struct Probe
    {
        const char *what;
        std::array<double, 3> local;
        bool hit;
    };

    // Cylinder r 0.05 m, length 0.2 m; probe sphere r 0.01 m. Each pair sits 1 mm either side of
    // contact. The last two probes are inside the capsule of the same radius but clear of the cylinder.
    constexpr double R = 0.05, L = 0.2, S = 0.01;
    const Probe probes[] = {
        {"centre on the axis", {0, 0, 0.1}, true},
        {"inside, off axis", {0.03, -0.02, 0.15}, true},
        {"mantle, just inside contact", {R + S - 0.001, 0, 0.1}, true},
        {"mantle, just clear", {R + S + 0.001, 0, 0.1}, false},
        {"mantle, diagonal, just clear", {(R + S + 0.001) / std::sqrt(2.0), (R + S + 0.001) / std::sqrt(2.0), 0.05}, false},
        {"top face, just inside contact", {0.02, 0.01, L + S - 0.001}, true},
        {"top face, just clear", {0.02, 0.01, L + S + 0.001}, false},
        {"bottom face, just inside contact", {0, -0.03, -S + 0.001}, true},
        {"bottom face, just clear", {0, -0.03, -S - 0.001}, false},
        {"top edge, just inside contact", {R + (S - 0.001) / std::sqrt(2.0), 0, L + (S - 0.001) / std::sqrt(2.0)}, true},
        {"top edge, just clear", {R + (S + 0.001) / std::sqrt(2.0), 0, L + (S + 0.001) / std::sqrt(2.0)}, false},
        {"bottom edge, just clear", {0, -R - (S + 0.001) / std::sqrt(2.0), -(S + 0.001) / std::sqrt(2.0)}, false},
        {"inside capsule cap, beyond the edge", {0.005, 0.0525, L + 0.015}, false},
        {"inside capsule cap, beyond the face", {0.04, 0, L + 0.03}, false},
    };

    void boundary_cases()
    {
        const Frame frames[] = {
            rotation(0, 0, 1, 0.0, {0, 0, 0}),
            rotation(1, 0, 0, M_PI / 2, {0.4, -0.3, 0.8}),
            rotation(0.3, -0.7, 0.2, 2.1, {-1.2, 0.9, 0.35}),
            rotation(-0.5, 0.5, 0.7, -0.9, {2.5, 1.5, -0.4}),
        };
        for (const auto &f : frames)
        {
            const auto c = make(f, R, L);
            for (const auto &p : probes)
            {
                const auto v = value(c, f.apply(p.local), S);
                check((v < 0.F) == p.hit, p.what);
                // the reference agrees with the expectation, so the probe itself is valid
                check((reference_distance(p.local, R, L) < S) == p.hit, p.what);
            }
        }
    }

    // Near contact, far from the origin, along a long axis: single precision must still classify
    // every probe whose true gap exceeds 50 um. On AVX `sqrt` is v * rsqrt(v), about 4e-4 relative,
    // which puts ~20 um of error on the radial distance at these radii.
    void precision_near_contact()
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        const double r = 0.042, length = 1.5, s = 0.03;
        int wrong = 0, total = 0;
        for (int k = 0; k < 20; ++k)
        {
            const auto f = rotation(u(rng), u(rng), u(rng), 3.0 * u(rng), {3.0 * u(rng), 3.0 * u(rng), 1.0 + u(rng)});
            const auto c = make(f, r, length);
            for (int i = 0; i < 500; ++i)
            {
                const double gap = (i % 2 ? 1.0 : -1.0) * (5e-5 + 4.5e-4 * std::abs(u(rng)));
                const double phi = M_PI * u(rng);
                const double z = length * (0.5 + 0.5 * u(rng));
                const double rho = r + s + gap;
                const std::array<double, 3> local{rho * std::cos(phi), rho * std::sin(phi), z};
                const bool hit = reference_distance(local, r, length) < s;
                wrong += (value(c, f.apply(local), s) < 0.F) != hit;
                ++total;
            }
        }
        if (wrong)
        {
            std::printf("near contact: %d of %d misclassified\n", wrong, total);
        }
        check(wrong == 0, "near-contact classification in single precision");
    }

    // The environment check reports a cylinder, and only where the cylinder is.
    void environment_check()
    {
        vamp::collision::Environment<float> e;
        e.cylinders.emplace_back(make(rotation(0, 1, 0, M_PI / 2, {0.5, 0.0, 0.3}), R, L));
        e.sort();
        const vamp::collision::Environment<V> ev(e);
        const auto in = [&](std::array<double, 3> p)
        {
            return vamp::sphere_environment_in_collision(
                ev, V(float(p[0])), V(float(p[1])), V(float(p[2])), V(float(S)));
        };
        // the axis runs along +x from (0.5, 0, 0.3) to (0.7, 0, 0.3)
        check(in({0.6, 0.0, 0.3}), "environment: sphere on the axis");
        check(in({0.6, 0.0, 0.3 + R + S - 0.001}), "environment: sphere touching the mantle");
        check(not in({0.6, 0.0, 0.3 + R + S + 0.001}), "environment: sphere clear of the mantle");
        check(not in({0.7 + 0.03, 0.0, 0.3 + 0.04}), "environment: sphere inside the capsule cap only");
    }
}  // namespace

int main()
{
    boundary_cases();
    precision_near_contact();
    environment_check();
    if (failures)
    {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("sphere_cylinder: all checks passed\n");
    return 0;
}
