// Ghi JSON tuan tu (thut le 2, giong json.dumps(indent=2) cua ban Python). Cho lanh: ghi result.json cuoi phien.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace uavp {

class JsonWriter {
public:
    JsonWriter& begin_object() { return open('{'); }
    JsonWriter& end_object() { return close('}'); }
    JsonWriter& begin_array() { return open('['); }
    JsonWriter& end_array() { return close(']'); }

    JsonWriter& key(std::string_view k) {
        separator();
        quoted(k);
        out_ += ": ";
        after_key_ = true;
        return *this;
    }
    JsonWriter& null() { return raw("null"); }
    JsonWriter& value(bool b) { return raw(b ? "true" : "false"); }
    JsonWriter& value(int v) { return raw(std::to_string(v)); }
    JsonWriter& value(int64_t v) { return raw(std::to_string(v)); }
    JsonWriter& value(std::size_t v) { return raw(std::to_string(v)); }
    JsonWriter& value(std::string_view s) {
        prefix();
        quoted(s);
        return *this;
    }
    JsonWriter& value(const char* s) { return value(std::string_view(s)); }
    JsonWriter& value(double v) {
        if (!std::isfinite(v)) return null();
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return raw(buf);
    }
    // So lam tron 2 chu so, NaN -> null (giong num() cua ban Python).
    JsonWriter& num(double v) { return std::isnan(v) ? null() : value(std::round(v * 100.0) / 100.0); }

    [[nodiscard]] const std::string& str() const noexcept { return out_; }

private:
    JsonWriter& open(char c) {
        prefix();
        out_ += c;
        first_.push_back(true);
        return *this;
    }
    JsonWriter& close(char c) {
        const bool empty = first_.back();
        first_.pop_back();
        if (!empty) newline();
        out_ += c;
        return *this;
    }
    JsonWriter& raw(std::string_view text) {
        prefix();
        out_ += text;
        return *this;
    }
    void prefix() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        separator();
    }
    void separator() {
        if (first_.empty()) return;
        if (!first_.back()) out_ += ',';
        first_.back() = false;
        newline();
    }
    void newline() {
        out_ += '\n';
        out_.append(2 * first_.size(), ' ');
    }
    void quoted(std::string_view s) {
        out_ += '"';
        for (const char c : s) {
            if (c == '"' || c == '\\') out_ += '\\';
            if (c == '\n') {
                out_ += "\\n";
                continue;
            }
            out_ += c;
        }
        out_ += '"';
    }

    std::string out_;
    std::vector<bool> first_;
    bool after_key_{false};
};

}  // namespace uavp
