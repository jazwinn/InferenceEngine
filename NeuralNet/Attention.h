#pragma once

#include "Math/Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "Activation.h"

namespace nn {

    using activation::Softmax;
    template <typename T>
    class Attention : public nn::Module<T> {
    public:
        Attention(size_t d_model, size_t num_heads, bool use_bias = true) :
            d_model(d_model),
            num_heads(num_heads),
            head_dim(d_model / num_heads),
            scale(std::sqrt(static_cast<T>(head_dim))),
            query(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
            key(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
            value(std::make_shared<Linear<T>>(d_model, d_model, use_bias)),
            output_proj(std::make_shared<Linear<T>>(d_model, d_model, use_bias)){

            this->RegisterSubmodule("query", query);
            this->RegisterSubmodule("key", key);
            this->RegisterSubmodule("value", value);
            this->RegisterSubmodule("output_proj", output_proj);
        }

        Tensor<T> SplitHeads(const Tensor<T>& x) {
            std::vector<size_t> new_shape = {x.Shape()[0], x.Shape()[1], num_heads, head_dim};
            return x.Reshape(new_shape).Transpose(1, 2).Contiguous(); // (batch_size, num_heads, seq_length, head_dim)
        }

        Tensor<T> MergeHeads(const Tensor<T>& x) {
            Tensor<T> transposed = x.Transpose(1, 2); // (batch_size, seq_length, num_heads, head_dim)
            return transposed.Reshape(transposed.Shape()[0], transposed.Shape()[1], d_model); // (batch_size, seq_length, d_model)
        }

        Tensor<T> Forward(const Tensor<T>& input) override {
            Tensor<T> Q = SplitHeads((*query)(input));
            Tensor<T> K = SplitHeads((*key)(input));
            Tensor<T> V = SplitHeads((*value)(input));

            Tensor<T> scores = MatMul(Q, K.Transpose(-2, -1)) / scale;
            Tensor<T> attention_weights = Softmax<T>(-1)(scores);
            Tensor<T> context = MatMul(attention_weights, V);
            Tensor<T> output = (*output_proj)(MergeHeads(context));

            return output;
        }

        std::shared_ptr<Linear<T>> query;
        std::shared_ptr<Linear<T>> key;
        std::shared_ptr<Linear<T>> value;
        std::shared_ptr<Linear<T>> output_proj;
        
        size_t d_model;
        size_t num_heads;
        size_t head_dim;
        T scale;


    };

}


