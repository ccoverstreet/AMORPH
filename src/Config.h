#ifndef AMORPH_Config_h
#define AMORPH_Config_h

#include <string>
#include <tuple>

namespace AMORPH
{

class Config
{
    private:

        size_t num_threads;

        std::string data_file;
        std::string dnest4_options_file;

        std::tuple<double, double> control_points;

        size_t max_num_narrow_peaks;
        size_t max_num_wide_peaks;

        double left_edge, right_edge;

        // Mock "command line options"
        int argc;
        char** argv;

    public:

        Config();
        ~Config();
        Config(const Config& other) = delete;
        Config(Config&& other) = delete;
        Config& operator = (const Config& other) = delete;

        // Load from YAML file
        void load(const char* filename);

        // Getters
        const std::string& get_data_file() const { return data_file; }
        int get_argc() const { return argc; }
        char** get_argv() const { return argv; }
        const std::tuple<double, double> get_control_points() const
        { return control_points; }
        size_t get_max_num_narrow_peaks() const
        { return max_num_narrow_peaks; }
        size_t get_max_num_wide_peaks() const
        { return max_num_wide_peaks; }
        double get_left_edge() const { return left_edge; }
        double get_right_edge() const { return right_edge; }

        // A global instance
        static Config global;

};

} // namespace AMORPH

#endif

