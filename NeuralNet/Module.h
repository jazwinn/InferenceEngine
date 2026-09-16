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
        virtual Tensor<T> forward(const Tensor<T>& input) = 0;

        Tensor<T> operator()(const Tensor<T>& input) {
            return forward(input);
        }
        
        void register_parameter(const std::string& name, std::shared_ptr<Tensor<T>> parameter){
            m_parameters[name] = parameter;
        }

        void register_submodule(const std::string& name, std::shared_ptr<Module<T>> submodule){
            m_submodules[name] = submodule;
        }

        std::vector<std::shared_ptr<Tensor<T>>> parameters() const {
            std::vector<std::shared_ptr<Tensor<T>>> params;
            for (const auto& [name, param] : m_parameters) {
                params.push_back(param);
            }
            for (const auto& [name, submodule] : m_submodules) {
                auto sub_params = submodule->parameters();
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }
            return params;
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<Tensor<T>>> m_parameters;
        std::unordered_map<std::string, std::shared_ptr<Module<T>>> m_submodules;
    };
}

