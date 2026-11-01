#pragma once

#include <vamp/collision/shapes.hh>
#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // Squared distance from the sphere's centre to the solid cylinder, minus the squared sphere
    // radius: negative where they overlap. Axial distance is counted beyond the flat ends, radial
    // distance beyond the mantle; inside the cylinder both are zero. Needs rdv = 1 / |axis|^2 of a
    // non-degenerate axis; cylinders are validated where they are built.
    template <typename DataT>
    inline constexpr auto sphere_cylinder(
        const Cylinder<DataT> &c,
        const DataT &x,
        const DataT &y,
        const DataT &z,
        const DataT &r) noexcept -> DataT
    {
        constexpr float far = 1e30F;
        auto dx = x - c.x1;
        auto dy = y - c.y1;
        auto dz = z - c.z1;
        auto dot = dot_3(dx, dy, dz, c.xv, c.yv, c.zv);
        auto t = dot * c.rdv;
        auto len2 = dot_3(c.xv, c.yv, c.zv, c.xv, c.yv, c.zv);
        auto axial = (t * -1.F).clamp(0.F, far) + (t - 1.F).clamp(0.F, far);
        auto axial2 = axial * axial * len2;
        // The perpendicular from the axis, taken directly rather than as |d|^2 - (d.v)^2/|v|^2, which
        // subtracts two nearly equal squares near a contact.
        auto px = dx - c.xv * t;
        auto py = dy - c.yv * t;
        auto pz = dz - c.zv * t;
        auto radial = (sqrt(dot_3(px, py, pz, px, py, pz)) - c.r).clamp(0.F, far);
        return axial2 + radial * radial - r * r;
    }

    template <typename DataT>
    inline constexpr auto sphere_cylinder(const Cylinder<DataT> &c, const Sphere<DataT> &s) noexcept -> DataT
    {
        return sphere_cylinder(c, s.x, s.y, s.z, s.r);
    }
}  // namespace vamp::collision
