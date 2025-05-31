#pragma once

#include <eigen3/Eigen/Dense>

#include "impl/joint.hpp"

namespace fast_tf {
namespace internal {

template <typename T>
struct get_depth {
    static constexpr int value = get_depth<typename Joint<T>::Parent>::value + 1;
};

template <>
struct get_depth<Null> {
    static constexpr int value = 0;
};

template <typename T>
constexpr int get_depth_v = get_depth<T>::value;

template <int depth1, int depth2, typename T1, typename T2>
constexpr auto get_lca(const T1& link1, const T2& link2) {
    if constexpr (std::is_same<T1, T2>::value)
        return T1{};
    else {
        if constexpr (depth1 > depth2)
            return get_lca<depth1 - 1, depth2>(typename Joint<T1>::Parent{}, link2);
        else if constexpr (depth2 > depth1)
            return get_lca<depth1, depth2 - 1>(link1, typename Joint<T2>::Parent{});
        else
            return get_lca<depth1 - 1, depth2 - 1>(
                typename Joint<T1>::Parent{}, typename Joint<T2>::Parent{});
    }
}

template <int, int>
constexpr auto get_lca(const Null&, const auto&) {
    return Null{};
}

template <int, int>
constexpr auto get_lca(const auto&, const Null&) {
    return Null{};
}

template <int, int>
constexpr auto get_lca(const Null&, const Null&) {
    return Null{};
}

template <typename T1, typename T2>
constexpr auto get_lca() {
    constexpr int depth1 = get_depth_v<T1>;
    constexpr int depth2 = get_depth_v<T2>;
    return get_lca<depth1, depth2>(T1{}, T2{});
}

template <typename T1, typename T2>
using get_lca_v = decltype(fast_tf::internal::get_lca<T1, T2>());

template <typename LcaT, typename LinkT, typename... JointCollectionTs>
requires(has_joint<LcaT, LinkT>)
inline auto accumulate_transform(JointCollectionTs&... collections) {
    return get_transform<LcaT, LinkT>(collections...);
}

template <typename LcaT, typename LinkT, typename... JointCollectionTs>
inline auto accumulate_transform(JointCollectionTs&... collections) {
    return accumulate_transform<LcaT, typename Joint<LinkT>::Parent>(collections...)
         * get_transform<typename Joint<LinkT>::Parent, LinkT>(collections...);
}

template <typename T>
concept is_position_of_link =
    std::is_convertible_v<T, typename Link<typename T::LinkType>::Position>;
template <typename T>
concept is_direction_vector_of_link =
    std::is_convertible_v<T, typename Link<typename T::LinkType>::DirectionVector>;
template <typename T>
concept is_rotation_of_link =
    std::is_convertible_v<T, typename Link<typename T::LinkType>::Rotation>;

template <internal::is_transform TransformT>
inline auto apply_rotation(const TransformT& transform, const auto& value) {
    return transform.linear() * value;
}

template <internal::is_rotation RotationT>
inline auto apply_rotation(const RotationT& rotation, const auto& value) {
    return rotation * value;
}

template <internal::is_translation TranslationT>
inline auto apply_rotation(const TranslationT&, const auto& value) {
    return value;
}

} // namespace internal

template <
    internal::is_link To, internal::is_position_of_link PositionT, typename... JointCollectionTs>
inline typename To::Position
    cast(const PositionT& position, const JointCollectionTs&... collections) {
    using From = PositionT::LinkType;
    if constexpr (std::is_same_v<From, To>) {
        return position;
    } else {
        using Lca = internal::get_lca_v<From, To>;
        if constexpr (std::is_same_v<Lca, From>) {
            return typename To::Position{
                internal::accumulate_transform<Lca, To>(collections...).inverse() * (*position)};
        } else if constexpr (std::is_same_v<Lca, To>) {
            return typename To::Position{
                internal::accumulate_transform<Lca, From>(collections...) * (*position)};
        } else {
            return typename To::Position{
                internal::accumulate_transform<Lca, To>(collections...).inverse()
                * internal::accumulate_transform<Lca, From>(collections...) * (*position)};
        }
    }
}

template <
    internal::is_link To, internal::is_direction_vector_of_link VectorT,
    typename... JointCollectionTs>
inline typename To::DirectionVector
    cast(const VectorT& rotation, const JointCollectionTs&... collections) {
    using From = VectorT::LinkType;
    if constexpr (std::is_same_v<From, To>) {
        return rotation;
    } else {
        using Lca = internal::get_lca_v<From, To>;
        if constexpr (std::is_same_v<Lca, From>) {
            return typename To::DirectionVector{internal::apply_rotation(
                internal::accumulate_transform<Lca, To>(collections...).inverse(), *rotation)};
        } else if constexpr (std::is_same_v<Lca, To>) {
            return typename To::DirectionVector{internal::apply_rotation(
                internal::accumulate_transform<Lca, From>(collections...), *rotation)};
        } else {
            return typename To::DirectionVector{internal::apply_rotation(
                internal::accumulate_transform<Lca, To>(collections...).inverse()
                    * internal::accumulate_transform<Lca, From>(collections...),
                *rotation)};
        }
    }
}

template <
    internal::is_link To, internal::is_rotation_of_link RotationT, typename... JointCollectionTs>
inline typename To::Rotation
    cast(const RotationT& rotation, const JointCollectionTs&... collections) {
    using From = RotationT::LinkType;
    if constexpr (std::is_same_v<From, To>) {
        return rotation;
    } else {
        using Lca = internal::get_lca_v<From, To>;
        if constexpr (std::is_same_v<Lca, From>) {
            return typename To::Rotation{internal::apply_rotation(
                internal::accumulate_transform<Lca, To>(collections...).inverse(), *rotation)};
        } else if constexpr (std::is_same_v<Lca, To>) {
            return typename To::Rotation{internal::apply_rotation(
                internal::accumulate_transform<Lca, From>(collections...), *rotation)};
        } else {
            return typename To::Rotation{internal::apply_rotation(
                internal::accumulate_transform<Lca, To>(collections...).inverse()
                    * internal::accumulate_transform<Lca, From>(collections...),
                *rotation)};
        }
    }
}

template <internal::is_link From, internal::is_link To, typename... JointCollectionTs>
auto lookup_transform(const JointCollectionTs&... collections) {
    if constexpr (std::is_same_v<From, To>) {
        return;
    } else {
        using Lca = internal::get_lca_v<From, To>;
        if constexpr (std::is_same_v<Lca, From>) {
            return internal::accumulate_transform<Lca, To>(collections...);
        } else if constexpr (std::is_same_v<Lca, To>) {
            return internal::accumulate_transform<Lca, From>(collections...).inverse();
        } else {
            return internal::accumulate_transform<Lca, From>(collections...).inverse()
                 * internal::accumulate_transform<Lca, To>(collections...);
        }
    }
}

} // namespace fast_tf