#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Math/Tensor.h"

namespace nn{

    using math::Tensor;
    
    template <typename T>
    class Module{
    public:
        virtual ~Module() = default;

        void RegisterParameter(const std::string& name, std::shared_ptr<Tensor<T>> parameter){
            m_parameters[name] = parameter;
        }

        void RegisterSubmodule(const std::string& name, std::shared_ptr<Module<T>> submodule){
            m_submodules[name] = submodule;
        }

        std::vector<std::shared_ptr<Tensor<T>>> Parameters() const {
            std::vector<std::shared_ptr<Tensor<T>>> params;
            for (const auto& [name, param] : m_parameters) {
                params.push_back(param);
            }
            for (const auto& [name, submodule] : m_submodules) {
                auto sub_params = submodule->Parameters();
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }
            return params;
        }
        

    private:
        std::unordered_map<std::string, std::shared_ptr<Tensor<T>>> m_parameters;
        std::unordered_map<std::string, std::shared_ptr<Module<T>>> m_submodules;
    };
}

