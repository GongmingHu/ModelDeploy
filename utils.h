#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <vector>
using namespace std;

namespace util
{

    struct ClassificationResults
    {
        vector<float> score;
        vector<std::string> label;
    };

    template <typename T>
    std::vector<float> softmax(const T *logits, unsigned int _size, unsigned int &max_id);
    

    template <typename T>
    std::vector<float> softmax(const std::vector<T> &logits, unsigned int &max_id);

    template <typename T>
    std::vector<unsigned int> argsort(const std::vector<T> &arr);
    

    template <typename T>
    std::vector<unsigned int> argsort(const T *arr, unsigned int _size);
    

}

template <typename T>
std::vector<float> util::softmax(const T *logits, unsigned int _size, unsigned int &max_id)
{
    if (_size == 0 || logits == nullptr)
        return {};
    float max_prob = 0.f, total_exp = 0.f;
    std::vector<float> softmax_probs(_size);
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = std::exp((float)logits[i]);
        total_exp += softmax_probs[i];
    }
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = softmax_probs[i] / total_exp;
        if (softmax_probs[i] > max_prob)
        {
            max_id = i;
            max_prob = softmax_probs[i];
        }
    }
    return softmax_probs;
}

template <typename T>
std::vector<float> util::softmax(const std::vector<T> &logits, unsigned int &max_id)
{
    if (logits.empty())
        return {};
    const unsigned int _size = logits.size();
    float max_prob = 0.f, total_exp = 0.f;
    std::vector<float> softmax_probs(_size);
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = std::exp((float)logits[i]);
        total_exp += softmax_probs[i];
    }
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = softmax_probs[i] / total_exp;
        if (softmax_probs[i] > max_prob)
        {
            max_id = i;
            max_prob = softmax_probs[i];
        }
    }
    return softmax_probs;
}

template <typename T>
std::vector<unsigned int> util::argsort(const std::vector<T> &arr)
{
    if (arr.empty())
        return {};
    const unsigned int _size = arr.size();
    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i < _size; ++i)
        indices.push_back(i);
    std::sort(indices.begin(), indices.end(),
                [&arr](const unsigned int a, const unsigned int b)
                { return arr[a] > arr[b]; });
    return indices;
}

template <typename T>
std::vector<unsigned int> util::argsort(const T *arr, unsigned int _size)
{
    if (_size == 0 || arr == nullptr)
        return {};
    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i < _size; ++i)
        indices.push_back(i);
    std::sort(indices.begin(), indices.end(),
                [arr](const unsigned int a, const unsigned int b)
                { return arr[a] > arr[b]; });
    return indices;
}

#endif