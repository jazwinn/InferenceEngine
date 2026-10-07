#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Tensor/Tensor.h"

namespace nn{

    using tensor::Tensor;
    using tensor::Device;

    template <typename T>
    class Module{
    public:
        using value_type = T;
        struct NamedParameter{
            std::string name;
            std::shared_ptr<Tensor<T>> tensor;
        };

        virtual ~Module() = default;

        void To(Device device){
            for (const auto& [name, param] : m_parameters) {
                if(param){
                    *param = param->To(device);
                }
            }
            for (const auto& [name, submodule] : m_submodules) {
                if(submodule){
                    submodule->To(device);
                }
            }
        }

        void RegisterParameter(const std::string& name, std::shared_ptr<Tensor<T>> parameter){
            Register(m_parameters, name, std::move(parameter));
        }

        void RegisterSubmodule(const std::string& name, std::shared_ptr<Module<T>> submodule){
            Register(m_submodules, name, std::move(submodule));
        }

        std::vector<NamedParameter> NamedParameters(const std::string& prefix = "") const {
            std::vector<NamedParameter> named;
            for (const auto& [name, param] : m_parameters) {
                named.emplace_back(prefix + name, param);
            }
            for (const auto& [name, submodule] : m_submodules) {
                auto sub_named = submodule->NamedParameters(prefix + name + ".");
                named.insert(named.end(), sub_named.begin(), sub_named.end());
            }
            return named;
        }

        std::vector<std::shared_ptr<Tensor<T>>> Parameters() const {
            std::vector<std::shared_ptr<Tensor<T>>> params;
            for (auto& [name, param] : NamedParameters()) {
                params.push_back(std::move(param));
            }
            return params;
        }

    private:
        template <typename Value>
        static void Register(std::vector<std::pair<std::string, Value>>& entries, const std::string& name, Value value){
            auto existing = std::find_if(entries.begin(), entries.end(),
                                         [&](const auto& entry) { return entry.first == name; });
            if (existing != entries.end()) {
                existing->second = std::move(value);
            } else {
                entries.emplace_back(name, std::move(value));
            }
        }

        std::vector<std::pair<std::string, std::shared_ptr<Tensor<T>>>> m_parameters;
        std::vector<std::pair<std::string, std::shared_ptr<Module<T>>>> m_submodules;
    };
}
