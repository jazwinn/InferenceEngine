#pragma once

#include "Math/Tensor.h"
#include "Module.h"

namespace nn {
    template <typename T>
    class Attention : public nn::Module<T> {
    public:
        Attention(size_t d_model, size_t num_heads, bool use_bias = true) :
            d_model(d_model),
            num_heads(num_heads),
            head_dim(d_model / num_heads),
            scale(std::sqrt(static_cast<T>(head_dim))) {

            query(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
            key(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
            value(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
            output(std::make_shared<Linear<T>>(d_model, d_model, use_bias)){

            this->RegisterSubmodule("query", query);
            this->RegisterSubmodule("key", key);
            this->RegisterSubmodule("value", value);
            this->RegisterSubmodule("output", output);
        }

        Tensor<T> SplitHeads(const Tensor<T>& x) {
            std::vector<size_t> new_shape = {x.Shape()[0], num_heads, head_dim};
            return x.Reshape(new_shape);
        }

        Tensor<T> Forward(const Tensor<T>& input) override {
            Tensor<T> Q = (*query)(input);
            Tensor<T> K = (*key)(input);
            Tensor<T> V = (*value)(input);

            Q = SplitHeads(Q);
            K = SplitHeads(K);
            V = SplitHeads(V);

            Tensor<T> scores = MatMul(Q, K.Transpose()) / scale;
            Tensor<T> attention_weights = Softmax(scores);
            return MatMul(attention_weights, V);
        }

        std::shared_ptr<Linear<T>> query;
        std::shared_ptr<Linear<T>> key;
        std::shared_ptr<Linear<T>> value;
        std::shared_ptr<Linear<T>> output;
        
        size_t d_model;
        size_t num_heads;
        size_t head_dim;
        T scale;


    };

}


