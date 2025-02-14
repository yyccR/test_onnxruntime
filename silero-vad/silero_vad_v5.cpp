
#include <iostream>
#include <string>
#include "memory"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "wav.h"


int silero_vad_onnxruntime_num_threads = 1;
std::string silero_vad_onnxruntime_model_path = "/Users/yang/CLionProjects/test_onnxruntime/silero-vad/silero_vad_v5.onnx";
// env变量要提到全局,在局部实例化会导致session后面调用run失败
std::unique_ptr<Ort::Env> silero_vad_onnxrutnime_env;
std::unique_ptr<Ort::Session> silero_vad_session;

std::vector<Ort::Value> ort_inputs;

std::vector<const char *> input_node_names = {"input", "state", "sr"};
std::vector<float> input;
unsigned int size_state = 2 * 1 * 128; // It's FIXED.
std::vector<float> _state;
std::vector<int64_t> sr;


int64_t input_node_dims[2] = {};
const int64_t state_node_dims[3] = {2, 1, 128};
const int64_t sr_node_dims[1] = {1};

// Outputs
std::vector<Ort::Value> ort_outputs;
std::vector<const char *> output_node_names = {"output", "stateN"};

/// 阈值越小表示越是噪声, 越大表示越是人声
float threshold = 0.5;
int sample_rate = 16000;
int sr_per_ms = sample_rate / 1000;

int window_size_samples = 32 * sr_per_ms;

int min_speech_samples = sr_per_ms * 32;
int speech_pad_samples = sr_per_ms * 32;
float min_silence_samples = sr_per_ms * 0;
float min_silence_samples_at_max_speech = sr_per_ms * 98;

float max_speech_samples = (
        sample_rate *  std::numeric_limits<float>::infinity()
        - window_size_samples
        - 2 * speech_pad_samples
        );


bool triggered = false;
unsigned int temp_end = 0;
unsigned int current_sample = 0;
// MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
int prev_end;
int next_start = 0;


#define __DEBUG_SPEECH_PROB___ 1


class timestamp_t
{
public:
    int start;
    int end;

    // default + parameterized constructor
    timestamp_t(int start = -1, int end = -1)
            : start(start), end(end)
    {
    };

    // assignment operator modifies object, therefore non-const
    timestamp_t& operator=(const timestamp_t& a)
    {
        start = a.start;
        end = a.end;
        return *this;
    };

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const timestamp_t& a) const
    {
        return (start == a.start && end == a.end);
    };
    std::string c_str()
    {
        //return std::format("timestamp {:08d}, {:08d}", start, end);
        return format("{start:%08d,end:%08d}", start, end);
    };
private:

    std::string format(const char* fmt, ...)
    {
        char buf[256];

        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);

        if (r < 0)
            // conversion failed
            return {};

        const size_t len = r;
        if (len < sizeof buf)
            // we fit in the buffer
            return { buf, len };

#if __cplusplus >= 201703L
        // C++17: Create a string and write to its underlying array
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);

        return s;
#else
        // C++11 or C++14: We need to allocate scratch memory
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);

        return { vbuf.get(), len };
#endif
    };
};

//Output timestamp
std::vector<timestamp_t> speeches;
timestamp_t current_speech;


int initOnnxruntimeSileroVAD(std::string& model_path, int num_threads) {

    silero_vad_onnxruntime_num_threads = num_threads;
    silero_vad_onnxruntime_model_path = model_path;

    if(silero_vad_session){
        silero_vad_session.reset();
        silero_vad_onnxrutnime_env.reset();
    }


    silero_vad_onnxrutnime_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "silero_vad_onnxrutnime_env");
    Ort::SessionOptions session_options;
//    OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options,NNAPI_FLAG_CPU_ONLY);
    session_options.SetInterOpNumThreads(silero_vad_onnxruntime_num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    silero_vad_session = std::make_unique<Ort::Session>(*silero_vad_onnxrutnime_env, silero_vad_onnxruntime_model_path.c_str(), session_options);

    return 0;
}

void destroyOnnxruntimeSileroVAD(){
    if(silero_vad_session){
        silero_vad_session.reset();
        silero_vad_onnxrutnime_env.reset();
    }
}

void predict(const std::vector<float> &data) {
    // Infer
    // Create ort tensors
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input.assign(data.begin(), data.end());
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            allocator_info, input.data(), input.size(), input_node_dims, 2);
    Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            allocator_info, _state.data(), _state.size(), state_node_dims, 3);
    Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            allocator_info, sr.data(), sr.size(), sr_node_dims, 1);

    // Clear and add inputs
    ort_inputs.clear();
    ort_inputs.emplace_back(std::move(input_ort));
    ort_inputs.emplace_back(std::move(state_ort));
    ort_inputs.emplace_back(std::move(sr_ort));

    // Infer
    ort_outputs = silero_vad_session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

    // Output probability & update h,c recursively
    float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
    float *stateN = ort_outputs[1].GetTensorMutableData<float>();
    std::memcpy(_state.data(), stateN, size_state * sizeof(float));

    // Push forward sample index
    current_sample += window_size_samples;


    // Reset temp_end when > threshold
    if ((speech_prob >= threshold))
    {
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample- window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        if (temp_end != 0)
        {
            temp_end = 0;
            if (next_start < prev_end)
                next_start = current_sample - window_size_samples;
        }
        if (triggered == false)
        {
            triggered = true;

            current_speech.start = current_sample - window_size_samples;
        }
        return;
    }

    if (
            (triggered == true)
            && ((current_sample - current_speech.start) > max_speech_samples)
            ) {
        if (prev_end > 0) {
            current_speech.end = prev_end;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();

            // previously reached silence(< neg_thres) and is still not speech(< thres)
            if (next_start < prev_end)
                triggered = false;
            else{
                current_speech.start = next_start;
            }
            prev_end = 0;
            next_start = 0;
            temp_end = 0;

        }
        else{
            current_speech.end = current_sample;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
        return;

    }
    if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold))
    {
        if (triggered) {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        }
        else {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        }
        return;
    }


    // 4) End
    if ((speech_prob < (threshold - 0.15)))
    {
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
        if (triggered == true)
        {
            if (temp_end == 0)
            {
                temp_end = current_sample;
            }
            if (current_sample - temp_end > min_silence_samples_at_max_speech)
                prev_end = temp_end;
            // a. silence < min_slience_samples, continue speaking
            if ((current_sample - temp_end) < min_silence_samples)
            {

            }
                // b. silence >= min_slience_samples, end speaking
            else
            {
                current_speech.end = temp_end;
                if (current_speech.end - current_speech.start > min_speech_samples)
                {
                    speeches.push_back(current_speech);
                    current_speech = timestamp_t();
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                }
            }
        }
        else {
            // may first windows see end state.
        }
        return;
    }

}

void vad()
{

    input.resize(window_size_samples);
    input_node_dims[0] = 1;
    input_node_dims[1] = window_size_samples;

    _state.resize(size_state);
    sr.resize(1);
    sr[0] = sample_rate;
    std::memset(_state.data(), 0.0f, _state.size() * sizeof(float));

    std::string model_path = "/Users/yang/CLionProjects/test_onnxruntime/silero-vad/silero_vad_v5.onnx";
    initOnnxruntimeSileroVAD(model_path, 1);
    // Read wav
    wav::WavReader wav_reader("/Users/yang/CLionProjects/test_onnxruntime/data/audio/test_chinese_1.wav"); //16000,1,32float
    std::vector<float> input_wav(wav_reader.num_samples());
    std::vector<float> output_wav;

    /// 数据复制
    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    }

    /// 分段预测
    int audio_length_samples = input_wav.size();
    for (int j = 0; j < audio_length_samples; j += window_size_samples)
    {
        if (j + window_size_samples > audio_length_samples)
            break;
        std::vector<float> r{ &input_wav[0] + j, &input_wav[0] + j + window_size_samples };
//        std::cout << audio_length_samples << " " << *(&input_wav[0] + j) << " " << *(&input_wav[0] + j + window_size_samples) << std::endl;
//        std::cout << j << std::endl;
        predict(r);
    }

    if (current_speech.start >= 0) {
        current_speech.end = audio_length_samples;
        speeches.push_back(current_speech);
        current_speech = timestamp_t();
        prev_end = 0;
        next_start = 0;
        temp_end = 0;
        triggered = false;
    }


    for (int i = 0; i < speeches.size(); i++) {

        std::cout << speeches[i].c_str() << std::endl;
    }


}


int test_silero_vad_v5() {

    vad();
    return 0;
};
