#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include <eigen3/Eigen/Dense>

#include "impl/joint.hpp"

namespace fast_tf {

namespace internal {

// https://stackoverflow.com/a/57528226
template <typename T, typename... Ts>
struct unique : std::type_identity<T> {};

template <typename... Ts, typename U, typename... Us>
struct unique<std::tuple<Ts...>, U, Us...>
    : std::conditional_t<
          (std::is_same_v<U, Ts> || ...), unique<std::tuple<Ts...>, Us...>,
          unique<std::tuple<Ts..., U>, Us...>> {};

template <typename... Ts>
using unique_tuple = typename unique<std::tuple<>, Ts...>::type;

// https://stackoverflow.com/a/41171291
template <typename T, typename U>
struct has_type;

template <typename T, typename... Us>
struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...> {};

template <typename T>
concept has_modification_getter = requires(T t, bool v) {
    { t.get_modified() } -> std::convertible_to<bool>;
};
inline bool get_joint_modified(const internal::has_modification_getter auto& joint) {
    return joint.get_modified();
}
inline constexpr bool get_joint_modified(const auto&) { return false; }

template <typename T>
concept has_modification_setter = requires(T t, bool v) {
    { t.set_modified(v) };
};
inline void set_joint_modified(internal::has_modification_setter auto& joint, bool value) {
    joint.set_modified(value);
}
inline constexpr void set_joint_modified(auto&, bool) {}

} // namespace internal

template <internal::is_non_root_link... ChildLinkTs>
class JointCollection {
public:
    using TupleT = internal::unique_tuple<Joint<ChildLinkTs>...>;

    JointCollection() requires(std::is_same_v<TupleT, std::tuple<Joint<ChildLinkTs>...>>) = default;

    [[deprecated("Link types passed in as template parameter is duplicated.")]]
    JointCollection() = default;

    template <typename ChildT>
    struct contains_joint {
        static constexpr bool value = internal::has_type<Joint<ChildT>, TupleT>::value;
    };

    template <typename ChildT>
    static constexpr bool contains_joint_v = contains_joint<ChildT>::value;

    template <internal::is_link From, internal::is_link To>
    requires internal::has_joint<From, To> auto& get_joint() {
        return std::get<fast_tf::Joint<To>>(collection_);
    }

    template <internal::is_link From, internal::is_link To>
    requires internal::has_joint<From, To> const auto& get_joint() const {
        return std::get<fast_tf::Joint<To>>(collection_);
    }

    template <internal::is_link From, internal::is_link To, typename... ArgTs>
    requires(internal::has_setter_joint<From, To, ArgTs...> && contains_joint_v<To>)
    void set_transform(ArgTs&&... value) {
        auto& joint = get_joint<From, To>();
        joint.set_transform(std::forward<ArgTs>(value)...);
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To, internal::is_transform TransformT>
    requires(
        !internal::has_setter_joint<From, To, TransformT> && internal::has_transform_joint<From, To>
        && contains_joint_v<To>)
    void set_transform(TransformT&& transform) {
        auto& joint     = get_joint<From, To>();
        joint.transform = std::forward<TransformT>(transform);
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To, internal::is_translation TranslationT>
    requires(
        !internal::has_setter_joint<From, To, TranslationT>
        && internal::has_translation_joint<From, To> && contains_joint_v<To>)
    void set_transform(TranslationT&& translation) {
        auto& joint     = get_joint<From, To>();
        joint.transform = std::forward<TranslationT>(translation);
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To, internal::is_translation TranslationT>
    requires(
        !internal::has_setter_joint<From, To, TranslationT>
        && internal::has_transform_joint<From, To> && contains_joint_v<To>)
    void set_transform(TranslationT&& translation) {
        auto& joint                   = get_joint<From, To>();
        joint.transform.translation() = std::forward<TranslationT>(translation).translation();
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To, internal::is_rotation RotationT>
    requires(
        !internal::has_setter_joint<From, To, RotationT> && internal::has_rotation_joint<From, To>
        && contains_joint_v<To>)
    void set_transform(RotationT&& rotation) {
        auto& joint     = get_joint<From, To>();
        joint.transform = std::forward<RotationT>(rotation);
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To, internal::is_rotation RotationT>
    requires(
        !internal::has_setter_joint<From, To, RotationT> && internal::has_transform_joint<From, To>
        && contains_joint_v<To>)
    void set_transform(RotationT&& rotation) {
        auto& joint              = get_joint<From, To>();
        joint.transform.linear() = std::forward<RotationT>(rotation).matrix();
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To, typename... ArgTs>
    requires(internal::has_stateful_joint<From, To, ArgTs...> && contains_joint_v<To>)
    void set_state(ArgTs&&... value) {
        auto& joint = get_joint<From, To>();
        joint.set_state(std::forward<ArgTs>(value)...);
        internal::set_joint_modified(joint, true);
    }

    template <internal::is_link From, internal::is_link To>
    requires(internal::has_getter_joint<From, To> && contains_joint_v<To>)
    auto get_transform() const {
        return get_joint<From, To>().get_transform();
    }

    template <internal::is_link From, internal::is_link To>
    requires(
        !internal::has_getter_joint<From, To> && internal::has_joint<From, To>
        && contains_joint_v<To>)
    const auto& get_transform() const {
        return get_joint<From, To>().transform;
    }

    template <typename JointCollectionTo>
    auto as_reference();

    template <typename F>
    requires requires(const F& f) {
        (f.template operator()<typename Joint<ChildLinkTs>::Parent, ChildLinkTs>(), ...);
    } static void for_each(const F& f) {
        (f.template operator()<typename Joint<ChildLinkTs>::Parent, ChildLinkTs>(), ...);
    }

    void set_all_unmodified() {
        for_each([this]<typename From, typename To>() {
            internal::set_joint_modified(get_joint<From, To>(), false);
        });
    }

    template <typename F>
    requires requires(const F& f) {
        (f.template operator()<typename Joint<ChildLinkTs>::Parent, ChildLinkTs>(), ...);
    } void for_each_modified(const F& f) {
        for_each([this, &f]<typename From, typename To>() {
            if (internal::get_joint_modified(get_joint<From, To>()))
                f.template operator()<From, To>();
        });
    }

private:
    TupleT collection_;
};

namespace internal {

template <internal::is_non_root_link... ChildLinkTs>
void test_joint_collection(const JointCollection<ChildLinkTs...>&);

template <typename T>
concept is_joint_collection = requires(T collection) { test_joint_collection(collection); };

template <
    internal::is_joint_collection FromCollection,
    internal::is_non_root_link... ToCollectionChildLinkTs>
requires(FromCollection::template contains_joint_v<ToCollectionChildLinkTs> && ...)
void test_joint_collection_castable(
    const FromCollection&, const JointCollection<ToCollectionChildLinkTs...>&);

template <typename From, typename To>
concept is_joint_collection_castable =
    requires(From from, To to) { test_joint_collection_castable(from, to); };

} // namespace internal

template <
    internal::is_joint_collection JointCollectionFrom,
    internal::is_joint_collection JointCollectionTo>
requires internal::is_joint_collection_castable<JointCollectionFrom, JointCollectionTo>
class JointCollectionReference {
public:
    explicit JointCollectionReference(JointCollectionFrom& ref)
        : ref_(ref) {}

    template <typename ChildT>
    static constexpr bool contains_joint_v =
        JointCollectionTo::template contains_joint<ChildT>::value;

    template <internal::is_link From, internal::is_link To, typename... ArgTs>
    requires(JointCollectionTo::template contains_joint_v<To> && requires(JointCollectionFrom collection, ArgTs&&... value) {
        collection.template set_transform<From, To>(std::forward<ArgTs>(value)...);
    }) void set_transform(ArgTs&&... value) {
        ref_.template set_transform<From, To>(std::forward<ArgTs>(value)...);
    }

    template <internal::is_link From, internal::is_link To, typename... ArgTs>
    requires(JointCollectionTo::template contains_joint_v<To> && requires(JointCollectionFrom collection, ArgTs&&... value) {
        collection.template set_state<From, To>(std::forward<ArgTs>(value)...);
    }) void set_state(ArgTs&&... value) {
        ref_.template set_state<From, To>(std::forward<ArgTs>(value)...);
    }

    template <internal::is_link From, internal::is_link To, typename... ArgTs>
    requires(JointCollectionTo::template contains_joint_v<To> && requires(JointCollectionFrom collection, ArgTs&&... value) {
        collection.template get_transform<From, To>(std::forward<ArgTs>(value)...);
    }) auto get_transform(ArgTs&&... value) const {
        return ref_.template get_transform<From, To>(std::forward<ArgTs>(value)...);
    }

    template <typename JointCollectionNext>
    requires internal::is_joint_collection_castable<JointCollectionTo, JointCollectionNext>
    auto as_reference() {
        return JointCollectionReference<JointCollectionFrom, JointCollectionNext>(ref_);
    }

    template <typename F>
    requires requires(const F& f) { JointCollectionTo::for_each(f); }
    static void for_each(const F& f) {
        JointCollectionTo::for_each(f);
    }

    void set_all_unmodified() {
        for_each([this]<typename From, typename To>() {
            internal::set_joint_modified(ref_.template get_joint<From, To>(), false);
        });
    }

    template <typename F>
    requires requires(const F& f) { JointCollectionTo::for_each(f); }
    void for_each_modified(const F& f) {
        for_each([this, &f]<typename From, typename To>() {
            if (internal::get_joint_modified(ref_.template get_joint<From, To>()))
                f.template operator()<From, To>();
        });
    }

private:
    JointCollectionFrom& ref_;
};

template <internal::is_non_root_link... ChildLinkTs>
template <typename JointCollectionTo>
auto JointCollection<ChildLinkTs...>::as_reference() {
    return JointCollectionReference<JointCollection<ChildLinkTs...>, JointCollectionTo>(*this);
}

} // namespace fast_tf