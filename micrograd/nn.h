#include "value.h"
#include <vector>
#include <random>
#include <cassert>

class Neuron{
    size_t nin;
    std::vector<Value> weights;
    Value bias;

    public:
        Neuron(size_t inputs): nin{inputs}, bias{0.0}{
            static std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            for (size_t i{0}; i<nin ;i++){weights.push_back(Value(dist(gen)));}
        };

        Value operator()(const std::vector<Value>& inputs) const{
            assert(inputs.size() == nin);
            Value output{bias};

            for (size_t i{0}; i<nin ;i++){output = output + (inputs[i] * weights[i]);}
            return output.tanh();
        }

        std::vector<Value> parameters() const{\
            std::vector<Value> params;
            params.reserve(nin+1);
            params.insert(params.end(), weights.begin(), weights.end());
            params.push_back(bias);
            return params;
        }
};

class Layer{
    std::vector<Neuron> neurons;
    size_t nin;
    size_t nout;

    public:
        Layer(size_t nin, size_t nout): nin{nin}, nout{nout}{
            for (size_t i{0}; i<nout ;i++){neurons.push_back(Neuron(nin));}
        }

        std::vector<Value> operator()(const std::vector<Value>& inputs) const {
            std::vector<Value> outs;
            outs.reserve(nout);
            for (const auto& neuron : neurons){
                outs.push_back(neuron(inputs));
            }
            return outs;
        }

        std::vector<Value> parameters() const {
            std::vector<Value> params;
            params.reserve(nout * (nin + 1)); 
            for (const auto& neuron : neurons) {
                auto n_params = neuron.parameters();
                params.insert(params.end(), n_params.begin(), n_params.end());
            }
            return params;
        }
};
