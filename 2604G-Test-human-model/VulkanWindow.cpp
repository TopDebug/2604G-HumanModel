
#include "VulkanWindow.h"

#include <QCoreApplication>
#include <QString>
#include <QString>
#include <QEventLoop>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QVulkanInstance>
#include <QWheelEvent>
#include <QWidget>
#include <QWindow>

#include <QtGlobal>

#include <algorithm>
#include <cmath>
#include <cstring>
#ifndef TEST_USE_PREBUILT_SPV_ONLY
#include <shaderc/shaderc.hpp>
#endif

VulkanWindow::VulkanWindow(QWidget* parent)
    : QWidget(parent) {
    _centralStack = new QWidget(this);
    _centralStack->setObjectName(QStringLiteral("centralRenderHost"));

    _vkWindow = new QWindow();
    _vkWindow->setSurfaceType(QWindow::VulkanSurface);

    _vkContainer = QWidget::createWindowContainer(_vkWindow, _centralStack);
    _vkContainer->setFocusPolicy(Qt::StrongFocus);
    // Without tracking, Qt omits MouseMove unless a button is held — breaks Ctrl/Shift/Alt + move.
    _vkContainer->setMouseTracking(true);
    _centralStack->setMouseTracking(true);
    setMouseTracking(true);
}

void VulkanWindow::setEmbeddedVulkanInstance(QVulkanInstance* inst) {
    _qVulkanInstance = inst;
    if (_vkWindow) {
        _vkWindow->setVulkanInstance(inst);
    }
}

namespace {
VkDeviceSize alignVkDeviceSize(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment < 1) {
        alignment = 1;
    }
    return (value + alignment - 1) / alignment * alignment;
}
} // namespace

static std::vector<char> readSPVFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

#ifndef TEST_USE_PREBUILT_SPV_ONLY
static std::vector<uint32_t> compileGLSL(
    const std::string& source,
    shaderc_shader_kind kind,
    const std::string& name) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    auto result = compiler.CompileGlslToSpv(source, kind, name.c_str(), options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error("Shader compile error (" + name + "):\n" + result.GetErrorMessage());
    }
    return std::vector<uint32_t>(result.cbegin(), result.cend());
}
#endif

#ifndef TEST_USE_PREBUILT_SPV_ONLY
static std::string readTextFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader source file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
}
#endif

namespace {
bool gUseValidation = ENABLE_VALIDATION;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

static VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

// 
//  Data structures
// 

// X = Red (1,0,0)  |  Y = Blue (0,0,1)  |  Z = Green (0,1,0)
// Each axis: shaft + two arrow-head lines
// For axis lines we still use Vertex type; normals set to zero
static const std::vector<Vertex> AXIS_VERTICES = {
    //  X  (Red) 
    {{ 0.00f,  0.00f,  0.00f}, {0.f,0.f,0.f}, {1.f, 0.f, 0.f}},
    {{ 1.00f,  0.00f,  0.00f}, {0.f,0.f,0.f}, {1.f, 0.f, 0.f}},

    //  Y  (Blue) 
    {{ 0.00f,  0.00f,  0.00f}, {0.f,0.f,0.f}, {0.f, 0.f, 1.f}},
    {{ 0.00f,  1.00f,  0.00f}, {0.f,0.f,0.f}, {0.f, 0.f, 1.f}},

    //  Z  (Green) 
    {{ 0.00f,  0.00f,  0.00f}, {0.f,0.f,0.f}, {0.f, 1.f, 0.f}},
    {{ 0.00f,  0.00f,  1.00f}, {0.f,0.f,0.f}, {0.f, 1.f, 0.f}},
};

static float computeNiceStep(float a, float b, int targetTicks) {
    const float span = std::fabs(b - a);
    if (span < 1e-6f) {
        return 1.0f;
    }
    const float raw = span / static_cast<float>(targetTicks);
    const float mag = std::pow(10.0f, std::floor(std::log10(raw)));
    const float n = raw / mag;
    float nice = 1.0f;
    if (n <= 1.0f) {
        nice = 1.0f;
    }
    else if (n <= 2.0f) {
        nice = 2.0f;
    }
    else if (n <= 5.0f) {
        nice = 5.0f;
    }
    else {
        nice = 10.0f;
    }
    return nice * mag;
}

static void appendLineSegment(
    std::vector<Vertex>& out,
    float x0, float y0,
    float x1, float y1,
    const glm::vec3& color) {
    const float zOverlay = -0.99f;
    const glm::vec3 n(0.0f, 0.0f, 0.0f);
    out.push_back(Vertex{ glm::vec3(x0, y0, zOverlay), n, color });
    out.push_back(Vertex{ glm::vec3(x1, y1, zOverlay), n, color });
}

static void appendGlyphSegments(
    std::vector<Vertex>& out,
    char ch,
    float ox, float oy,
    float w, float h,
    const glm::vec3& color) {
    // Ortho Y grows toward the viewport top; `oy` is the glyph baseline / cell bottom.
    const auto seg = [&](bool on, float x0, float y0, float x1, float y1) {
        if (on) {
            appendLineSegment(out, ox + x0 * w, oy + y0 * h, ox + x1 * w, oy + y1 * h, color);
        }
    };

    bool a = false, b = false, c = false, d = false, e = false, f = false, g = false;
    switch (ch) {
    case '0': a = b = c = d = e = f = true; break;
    case '1': b = c = true; break;
    case '2': a = b = d = e = g = true; break;
    case '3': a = b = c = d = g = true; break;
    case '4': b = c = f = g = true; break;
    case '5': a = c = d = f = g = true; break;
    case '6': a = c = d = e = f = g = true; break;
    case '7': a = b = c = true; break;
    case '8': a = b = c = d = e = f = g = true; break;
    case '9': a = b = c = d = f = g = true; break;
    case '-': g = true; break;
    case '.':
        // Decimal point drawn as ~4 px segment (char width is ~6 px).
        appendLineSegment(out, ox + 0.17f * w, oy + 0.02f * h, ox + 0.83f * w, oy + 0.02f * h, color);
        return;
    case ',':
        appendLineSegment(out, ox + 0.45f * w, oy + 0.02f * h, ox + 0.55f * w, oy + -0.10f * h, color);
        return;
    case '(':
        appendLineSegment(out, ox + 0.60f * w, oy + 1.00f * h, ox + 0.40f * w, oy + 0.50f * h, color);
        appendLineSegment(out, ox + 0.40f * w, oy + 0.50f * h, ox + 0.60f * w, oy + 0.00f * h, color);
        return;
    case ')':
        appendLineSegment(out, ox + 0.40f * w, oy + 1.00f * h, ox + 0.60f * w, oy + 0.50f * h, color);
        appendLineSegment(out, ox + 0.60f * w, oy + 0.50f * h, ox + 0.40f * w, oy + 0.00f * h, color);
        return;
    case ' ':
        return;
    default:
        return;
    }

    seg(a, 0.0f, 1.0f, 1.0f, 1.0f);
    seg(b, 1.0f, 1.0f, 1.0f, 0.5f);
    seg(c, 1.0f, 0.5f, 1.0f, 0.0f);
    seg(d, 0.0f, 0.0f, 1.0f, 0.0f);
    seg(e, 0.0f, 0.5f, 0.0f, 0.0f);
    seg(f, 0.0f, 1.0f, 0.0f, 0.5f);
    seg(g, 0.0f, 0.5f, 1.0f, 0.5f);
}

static std::string formatRulerValue(float value, float step) {
    // Avoid "-0" labels from floating-point rounding around zero.
    if (std::fabs(value) < 0.5f * step) {
        value = 0.0f;
    }
    int decimals = 0;
    if (step < 1.0f) {
        decimals = static_cast<int>(std::ceil(-std::log10(step)));
        if (decimals < 0) {
            decimals = 0;
        }
        if (decimals > 3) {
            decimals = 3;
        }
    }
    char buf[64] = {};
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, static_cast<double>(value));
    std::string s(buf);
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') {
            s.pop_back();
        }
        if (!s.empty() && s.back() == '.') {
            s.pop_back();
        }
    }
    if (s.empty()) {
        s = "0";
    }
    if (s == "-0") {
        s = "0";
    }
    return s;
}

static void appendLabelSegmentsCentered(
    std::vector<Vertex>& out,
    const std::string& text,
    float centerX, float baseY,
    float charW, float charH,
    float spacing,
    const glm::vec3& color) {
    if (text.empty()) {
        return;
    }
    const float totalW = static_cast<float>(text.size()) * charW +
        static_cast<float>(text.size() > 0 ? text.size() - 1 : 0) * spacing;
    float x = centerX - totalW * 0.5f;
    for (char ch : text) {
        appendGlyphSegments(out, ch, x, baseY, charW, charH, color);
        x += charW + spacing;
    }
}

static void appendLabelSegmentsRightAligned(
    std::vector<Vertex>& out,
    const std::string& text,
    float rightX, float centerY,
    float charW, float charH,
    float spacing,
    const glm::vec3& color) {
    if (text.empty()) {
        return;
    }
    const float totalW = static_cast<float>(text.size()) * charW +
        static_cast<float>(text.size() > 0 ? text.size() - 1 : 0) * spacing;
    float x = rightX - totalW;
    const float y = centerY - charH * 0.5f;
    for (char ch : text) {
        appendGlyphSegments(out, ch, x, y, charW, charH, color);
        x += charW + spacing;
    }
}

static void appendLabelSegmentsLeftAligned(
    std::vector<Vertex>& out,
    const std::string& text,
    float leftX, float baseY,
    float charW, float charH,
    float spacing,
    const glm::vec3& color) {
    if (text.empty()) {
        return;
    }
    float x = leftX;
    for (char ch : text) {
        appendGlyphSegments(out, ch, x, baseY, charW, charH, color);
        x += charW + spacing;
    }
}

static void appendVulkanRulerVertices(
    std::vector<Vertex>& out,
    const glm::vec4& ortho,
    float viewportWidth,
    float viewportHeight,
    float mouseX,
    float mouseY,
    bool hasDepthWorld,
    const glm::vec3& depthWorld) {
    const float L = ortho.x;
    const float R = ortho.y;
    const float B = ortho.z;
    const float T = ortho.w;
    const float zOverlay = -0.99f;
    const glm::vec3 n(0.0f, 0.0f, 0.0f);
    const glm::vec3 xColor(1.0f, 0.0f, 0.0f);
    const glm::vec3 yColor = xColor;
    const glm::vec3 textColor(0.95f, 0.95f, 0.95f);

    const float stepX = computeNiceStep(L, R, 8);
    const float stepY = computeNiceStep(B, T, 8);
    float commonStep = stepX;
    if (stepY > 0.0f && (commonStep <= 0.0f || stepY < commonStep)) {
        commonStep = stepY;
    }
    if (commonStep <= 0.0f) {
        commonStep = 1.0f;
    }

    const float xSpan = std::fabs(R - L);
    const float ySpan = std::fabs(T - B);
    const float safeW = (viewportWidth > 1.0f) ? viewportWidth : 1.0f;
    const float safeH = (viewportHeight > 1.0f) ? viewportHeight : 1.0f;
    const float worldPerPixelX = (xSpan > 0.0f) ? (xSpan / safeW) : 0.0f;
    const float worldPerPixelY = (ySpan > 0.0f) ? (ySpan / safeH) : 0.0f;

    // Keep visual sizes in fixed screen pixels.
    constexpr float kAxisInsetPx = 0.5f;       // pin to edge with pixel-center alignment
    constexpr float kTickLenPx = 5.0f;
    constexpr float kCharHeightPx = 10.0f;
    constexpr float kCharWidthPx = 6.0f;
    constexpr float kCharSpacingPx = 2.0f;

    // With Vulkan Y-flip in projection, visual bottom maps to world-space T.
    const float bottomY = B + kAxisInsetPx * worldPerPixelY;
    const float rightX = R - kAxisInsetPx * worldPerPixelX;

    // Bottom (X) and right (Y) rulers in world-space overlay.
    out.push_back(Vertex{ glm::vec3(L, bottomY, zOverlay), n, xColor });
    out.push_back(Vertex{ glm::vec3(R, bottomY, zOverlay), n, xColor });
    out.push_back(Vertex{ glm::vec3(rightX, B, zOverlay), n, yColor });
    out.push_back(Vertex{ glm::vec3(rightX, T, zOverlay), n, yColor });

    const float xTickLen = kTickLenPx * worldPerPixelY;
    const float yTickLen = kTickLenPx * worldPerPixelX;
    const float charH = kCharHeightPx * worldPerPixelY;
    const float charW = kCharWidthPx * worldPerPixelX;
    const float charSpacing = kCharSpacingPx * worldPerPixelX;
    const float bottomLabelY = bottomY + xTickLen + charH * 1.25f;
    const float rightLabelX = rightX - yTickLen - charW * 0.60f;

    const float xMin = (L < R) ? L : R;
    const float xMax = (L < R) ? R : L;
    float xMark = std::ceil(xMin / commonStep) * commonStep;
    for (int guard = 0; guard < 1000 && xMark <= xMax + commonStep * 0.5f; ++guard, xMark += commonStep) {
        out.push_back(Vertex{ glm::vec3(xMark, bottomY, zOverlay), n, xColor });
        out.push_back(Vertex{ glm::vec3(xMark, bottomY + xTickLen, zOverlay), n, xColor });
        const std::string text = formatRulerValue(xMark, commonStep);
        appendLabelSegmentsCentered(out, text, xMark, bottomLabelY, charW, charH, charSpacing, textColor);
    }

    const float yMin = (B < T) ? B : T;
    const float yMax = (B < T) ? T : B;
    float yMark = std::ceil(yMin / commonStep) * commonStep;
    for (int guard = 0; guard < 1000 && yMark <= yMax + commonStep * 0.5f; ++guard, yMark += commonStep) {
        out.push_back(Vertex{ glm::vec3(rightX, yMark, zOverlay), n, yColor });
        out.push_back(Vertex{ glm::vec3(rightX - yTickLen, yMark, zOverlay), n, yColor });
        // World Y at tick; matches glm::ortho(L,R,B,T) and orthoB/orthoT spins (no negation).
        const std::string text = formatRulerValue(yMark, commonStep);
        appendLabelSegmentsRightAligned(out, text, rightLabelX, yMark, charW, charH, charSpacing, textColor);
    }

}

void VulkanWindow::markOverlayDirty() {
    _overlayDirty = true;
}

void VulkanWindow::setHostTopLevelWindow(QWidget* host) {
    _hostTopLevelWindow = host;
}

void VulkanWindow::syncViewportFromEmbeddedWindow() {
    int w = 0;
    int h = 0;
    getQtWindowSize(w, h);
    if (w > 0 && h > 0) {
        _viewport = glm::ivec2(w, h);
    }
}

void VulkanWindow::attachQtWindow(QWindow* window, QVulkanInstance* vulkanInstance) {
    _qtWindow = window;
    _qVulkanInstance = vulkanInstance;
    _ownsVkInstance = false;
    _instance = vulkanInstance->vkInstance();
    loadViewConfig();
    // Loaded windowW/H are the top-level window size; embedded surface size is applied after layout.
    if (_viewport.x > 0 && _viewport.y > 0) {
        if (_hostTopLevelWindow != nullptr) {
            _hostTopLevelWindow->resize(_viewport.x, _viewport.y);
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
        else {
            _qtWindow->resize(_viewport.x, _viewport.y);
        }
    }
    syncViewportFromEmbeddedWindow();
    initVulkan();
}

void VulkanWindow::shutdownWindowSystem() {
    saveViewConfigIfChanged();
    cleanup();
}

void VulkanWindow::getQtWindowSize(int& outW, int& outH) const {
    if (!_qtWindow) {
        outW = 0;
        outH = 0;
        return;
    }
    outW = _qtWindow->width();
    outH = _qtWindow->height();
}

void VulkanWindow::getQtFramebufferSize(int& outW, int& outH) const {
    if (!_qtWindow) {
        outW = 0;
        outH = 0;
        return;
    }
    const qreal dpr = _qtWindow->devicePixelRatio();
    outW = static_cast<int>(std::lround(static_cast<qreal>(_qtWindow->width()) * dpr));
    outH = static_cast<int>(std::lround(static_cast<qreal>(_qtWindow->height()) * dpr));
}

void VulkanWindow::getRenderTargetPixelSize(int& outW, int& outH) const {
    if (_swapChainExtent.width > 0U && _swapChainExtent.height > 0U) {
        outW = static_cast<int>(_swapChainExtent.width);
        outH = static_cast<int>(_swapChainExtent.height);
        return;
    }
    getQtFramebufferSize(outW, outH);
}

bool VulkanWindow::handleQtEvent(QObject* watched, QEvent* event) {
    switch (event->type()) {
    case QEvent::Resize:
        // Embedded Vulkan: resize is usually delivered on the container widget, not the QWindow.
        if (watched == static_cast<QObject*>(_qtWindow)
            || watched == static_cast<QObject*>(_vkContainer)) {
            syncViewportFromEmbeddedWindow();
        }
        _framebufferResized = true;
        return false;
    case QEvent::MouseMove: {
        const auto* e = static_cast<QMouseEvent*>(event);
        const QPointF p = e->position();
        cursorPosCallbackImpl(p.x(), p.y());
        return false;
    }
    case QEvent::MouseButtonPress:
    case QEvent::MouseButtonRelease: {
        const auto* e = static_cast<QMouseEvent*>(event);
        mouseButtonCallbackImpl(e->button(), event->type(), e->modifiers(), e->position().x(), e->position().y(), e->timestamp());
        return false;
    }
    case QEvent::Wheel: {
        const auto* e = static_cast<QWheelEvent*>(event);
        const float yoff = static_cast<float>(e->angleDelta().y()) / 120.0f;
        scrollCallbackImpl(yoff);
        return false;
    }
    case QEvent::KeyPress: {
        return true;
    }
    default:
        return false;
    }
}

bool VulkanWindow::loadViewConfig() {
    const std::filesystem::path configPath = "config.xml";
    std::ifstream input(configPath);
    if (!input.is_open()) {
        return false;
    }

    const std::string xml((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());

    auto readFloat = [&](const char* tag, float& outValue) -> bool {
        const std::string openTag = std::string("<") + tag + ">";
        const std::string closeTag = std::string("</") + tag + ">";
        const std::size_t start = xml.find(openTag);
        if (start == std::string::npos) {
            return false;
        }
        const std::size_t valueBegin = start + openTag.size();
        const std::size_t end = xml.find(closeTag, valueBegin);
        if (end == std::string::npos) {
            return false;
        }
        try {
            outValue = std::stof(xml.substr(valueBegin, end - valueBegin));
            return true;
        }
        catch (...) {
            return false;
        }
    };

    float coordinateX = _coordinate.x;
    float coordinateY = _coordinate.y;
    float coordinateZ = _coordinate.z;
    float rotationX = _rotation.x;
    float rotationY = _rotation.y;
    float orthoL = _ortho.x;
    float orthoR = _ortho.y;
    float orthoB = _ortho.z;
    float orthoT = _ortho.w;
    float windowW = static_cast<float>(_viewport.x);
    float windowH = static_cast<float>(_viewport.y);

    const bool ok =
        readFloat("coordinateX", coordinateX) &&
        readFloat("coordinateY", coordinateY) &&
        readFloat("coordinateZ", coordinateZ) &&
        readFloat("rotationX", rotationX) &&
        readFloat("rotationY", rotationY) &&
        readFloat("orthoL", orthoL) &&
        readFloat("orthoR", orthoR) &&
        readFloat("orthoB", orthoB) &&
        readFloat("orthoT", orthoT) &&
        (readFloat("windowW", windowW) || readFloat("viewportW", windowW)) &&
        (readFloat("windowH", windowH) || readFloat("viewportH", windowH));

    if (!ok) {
        return false;
    }

    _coordinate = glm::vec3(coordinateX, coordinateY, coordinateZ);
    setRotation(glm::vec2(rotationX, rotationY));
    _ortho = glm::vec4(orthoL, orthoR, orthoB, orthoT);
    _viewport = glm::ivec2(windowW, windowH);
    _lastSavedCoordinate = _coordinate;
    _lastSavedRotation = _rotation;
    _lastSavedOrtho = _ortho;
    _lastSavedWindowSize = glm::ivec2(windowW, windowH);
    _hasSavedViewState = true;
    return true;
}

bool VulkanWindow::saveViewConfig() const {
    const std::filesystem::path configPath = "config.xml";
    std::ofstream output(configPath, std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }

    output << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    output << "<Config>\n";
    output << "    <View>\n";
    output << std::setprecision(9);
    output << "        <coordinateX>" << _coordinate.x << "</coordinateX>\n";
    output << "        <coordinateY>" << _coordinate.y << "</coordinateY>\n";
    output << "        <coordinateZ>" << _coordinate.z << "</coordinateZ>\n";
    output << "        <rotationX>" << _rotation.x << "</rotationX>\n";
    output << "        <rotationY>" << _rotation.y << "</rotationY>\n";
    output << "        <orthoL>" << _ortho.x << "</orthoL>\n";
    output << "        <orthoR>" << _ortho.y << "</orthoR>\n";
    output << "        <orthoB>" << _ortho.z << "</orthoB>\n";
    output << "        <orthoT>" << _ortho.w << "</orthoT>\n";
    glm::ivec2 winSz = getWindowSize();
    if (winSz.x < 1 || winSz.y < 1) {
        int x = 0;
        int y = 0;
        getQtWindowSize(x, y);
        winSz = glm::ivec2(std::max(1, x), std::max(1, y));
    }
    output << "        <windowW>" << winSz.x << "</windowW>\n";
    output << "        <windowH>" << winSz.y << "</windowH>\n";
    output << "    </View>\n";
    output << "</Config>\n";

    return output.good();
}

void VulkanWindow::saveViewConfigIfChanged() {
    syncViewportFromEmbeddedWindow();

    const glm::ivec2 mainLogical = getWindowSize();

    const bool changed =
        !_hasSavedViewState ||
        _coordinate != _lastSavedCoordinate ||
        _rotation != _lastSavedRotation ||
        _ortho != _lastSavedOrtho ||
        mainLogical != _lastSavedWindowSize;

    if (!changed) {
        return;
    }

    if (saveViewConfig()) {
        _lastSavedCoordinate = _coordinate;
        _lastSavedRotation = _rotation;
        _lastSavedOrtho = _ortho;
        _lastSavedWindowSize = mainLogical;
        _hasSavedViewState = true;
    }
}

void VulkanWindow::cleanup() {
    if (_device == VK_NULL_HANDLE) {
        return;
    }
    vkDeviceWaitIdle(_device);

    cleanupSwapChain();

    for (size_t i = 0; i < _uniformBuffers.size(); ++i) {
        vkDestroyBuffer(_device, _uniformBuffers[i], nullptr);
        vkFreeMemory(_device, _uniformBuffersMemory[i], nullptr);
    }
    vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(_device, _descriptorSetLayout, nullptr);
    vkDestroyBuffer(_device, _vertexBuffer, nullptr);
    vkFreeMemory(_device, _vertexBufferMemory, nullptr);
    vkDestroyPipeline(_device, _trianglePipeline, nullptr);
    vkDestroyPipeline(_device, _linePipeline, nullptr);
    vkDestroyPipeline(_device, _rulerPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);
    vkDestroyRenderPass(_device, _renderPass, nullptr);
    vkDestroyCommandPool(_device, _commandPool, nullptr);
    _commandPool = VK_NULL_HANDLE;

    const size_t syncCount = std::min({
        _imageAvailableSemaphores.size(),
        _renderFinishedSemaphores.size(),
        _inFlightFences.size()});
    for (size_t i = 0; i < syncCount; ++i) {
        vkDestroySemaphore(_device, _imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(_device, _renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(_device, _inFlightFences[i], nullptr);
    }
    _imageAvailableSemaphores.clear();
    _renderFinishedSemaphores.clear();
    _inFlightFences.clear();

    vkDestroyDevice(_device, nullptr);
    _device = VK_NULL_HANDLE;

    if (ENABLE_VALIDATION && _debugMessenger != VK_NULL_HANDLE) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) {
            fn(_instance, _debugMessenger, nullptr);
        }
        _debugMessenger = VK_NULL_HANDLE;
    }
    if (_surface != VK_NULL_HANDLE && _instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        _surface = VK_NULL_HANDLE;
    }
    if (_ownsVkInstance && _instance != VK_NULL_HANDLE) {
        vkDestroyInstance(_instance, nullptr);
        _instance = VK_NULL_HANDLE;
    }
}

glm::mat4 VulkanWindow::buildViewMatrix() const {
    glm::quat quatX = glm::angleAxis(glm::radians(-_rotation[0]), glm::vec3(1, 0, 0));
    glm::quat quatY = glm::angleAxis(glm::radians(-_rotation[1]), glm::vec3(0, 1, 0));
    glm::vec3 viewY = quatY * quatX * glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 viewZ = quatY * quatX * glm::vec3(0.0f, 0.0f, -1.0f);
    return glm::lookAt(_coordinate - viewZ * 100.0f, _coordinate + viewZ * 100.0f, viewY);
}

glm::mat4 VulkanWindow::buildProjMatrix() const {
    glm::mat4 proj = glm::ortho(_ortho[0], _ortho[1], _ortho[3], _ortho[2], 0.1f, 200.0f);
    return proj;
}

glm::vec4 VulkanWindow::buildRulerOrtho() const {
    glm::quat quatX = glm::angleAxis(glm::radians(-_rotation[0]), glm::vec3(1, 0, 0));
    glm::quat quatY = glm::angleAxis(glm::radians(-_rotation[1]), glm::vec3(0, 1, 0));

    // Screen-right / screen-up directions in world space (same basis as buildViewMatrix).
    const glm::vec3 viewX = quatY * quatX * glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 viewY = quatY * quatX * glm::vec3(0.0f, 1.0f, 0.0f);

    const float dx = glm::dot(_coordinate, viewX);
    const float dy = glm::dot(_coordinate, viewY);

    glm::vec4 rulerOrtho;
    rulerOrtho[0] = _ortho[0] + dx;
    rulerOrtho[1] = _ortho[1] + dx;
    rulerOrtho[2] = _ortho[2] + dy;
    rulerOrtho[3] = _ortho[3] + dy;

    return rulerOrtho;
}

bool VulkanWindow::hasStencilComponent(VkFormat format) const {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

bool VulkanWindow::readDepthAtFramebufferPixel(uint32_t px, uint32_t py, float& outDepth01) {
    if (_device == VK_NULL_HANDLE || _commandPool == VK_NULL_HANDLE || _depthImage == VK_NULL_HANDLE) {
        return false;
    }
    if (px >= _swapChainExtent.width || py >= _swapChainExtent.height) {
        return false;
    }

    // Keep this simple/safe for picking: block until current frame work is finished.
    vkDeviceWaitIdle(_device);

    const VkDeviceSize bpp = 4;
    const VkDeviceSize rowPitchBytes = alignVkDeviceSize(bpp, _copyRowPitchAlign);
    const VkDeviceSize stagingSize = alignVkDeviceSize(rowPitchBytes, _copyOffsetAlign);

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    createBuffer(
        stagingSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingMemory);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = _commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(_device, &allocInfo, &cmd) != VK_SUCCESS || cmd == VK_NULL_HANDLE) {
        vkDestroyBuffer(_device, stagingBuffer, nullptr);
        vkFreeMemory(_device, stagingMemory, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(_device, _commandPool, 1, &cmd);
        vkDestroyBuffer(_device, stagingBuffer, nullptr);
        vkFreeMemory(_device, stagingMemory, nullptr);
        return false;
    }

    // Temporarily switch depth image into transfer-src so we can copy one pixel.
    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    if (_depthImageGpuReady) {
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        toTransfer.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    else {
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.srcAccessMask = 0;
    }
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = _depthImage;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (hasStencilComponent(findDepthFormat())) {
        toTransfer.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    toTransfer.subresourceRange.baseMipLevel = 0;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount = 1;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        _depthImageGpuReady ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &toTransfer);

    // Copy exactly one depth pixel under the cursor.
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = static_cast<uint32_t>(rowPitchBytes / bpp);
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { static_cast<int32_t>(px), static_cast<int32_t>(py), 0 };
    region.imageExtent = { 1, 1, 1 };

    vkCmdCopyImageToBuffer(
        cmd,
        _depthImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        stagingBuffer,
        1,
        &region);

    // Restore the depth attachment layout for normal rendering.
    VkImageMemoryBarrier backToDepth{};
    backToDepth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    backToDepth.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    backToDepth.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    backToDepth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToDepth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToDepth.image = _depthImage;
    backToDepth.subresourceRange = toTransfer.subresourceRange;
    backToDepth.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    backToDepth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &backToDepth);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphicsQueue);

    void* mapped = nullptr;
    vkMapMemory(_device, stagingMemory, 0, stagingSize, 0, &mapped);
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(mapped);
    vkUnmapMemory(_device, stagingMemory);

    vkFreeCommandBuffers(_device, _commandPool, 1, &cmd);
    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingMemory, nullptr);

    // Decode copied depth based on the runtime-selected depth format.
    const VkFormat depthFmt = findDepthFormat();
    if (depthFmt == VK_FORMAT_D32_SFLOAT || depthFmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        outDepth01 = *reinterpret_cast<const float*>(&packed);
    }
    else if (depthFmt == VK_FORMAT_D24_UNORM_S8_UINT) {
        outDepth01 = static_cast<float>(packed & 0x00FFFFFFu) / 16777215.0f;
    }
    else {
        return false;
    }

    outDepth01 = std::clamp(outDepth01, 0.0f, 1.0f);
    // If depth is near clear value, no entity was drawn at this pixel.
    // Treat "almost 1.0" as invalid pick to avoid far-plane false positives.
    if (outDepth01 >= 0.999f) {
        return false;
    }
    return true;
}

bool VulkanWindow::readDepthBufferFloat(std::vector<float>& outDepth, uint32_t& outW, uint32_t& outH) {
    if (_device == VK_NULL_HANDLE || _depthImage == VK_NULL_HANDLE || _commandPool == VK_NULL_HANDLE) {
        return false;
    }
    const uint32_t w = _swapChainExtent.width;
    const uint32_t h = _swapChainExtent.height;
    if (w == 0 || h == 0) {
        return false;
    }

    vkDeviceWaitIdle(_device);

    const VkFormat depthFmt = findDepthFormat();
    const VkDeviceSize bytesPerPixel = 4;
    const VkDeviceSize rowPitchBytes =
        alignVkDeviceSize(static_cast<VkDeviceSize>(w) * bytesPerPixel, _copyRowPitchAlign);
    const VkDeviceSize bufferSize = rowPitchBytes * static_cast<VkDeviceSize>(h);

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    try {
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingMemory);
    } catch (...) {
        return false;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = _commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(_device, &allocInfo, &cmd) != VK_SUCCESS) {
        vkDestroyBuffer(_device, stagingBuffer, nullptr);
        vkFreeMemory(_device, stagingMemory, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    if (_depthImageGpuReady) {
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        toTransfer.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    else {
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.srcAccessMask = 0;
    }
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = _depthImage;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (hasStencilComponent(depthFmt)) {
        toTransfer.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    toTransfer.subresourceRange.baseMipLevel = 0;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount = 1;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        _depthImageGpuReady ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &toTransfer);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = static_cast<uint32_t>(rowPitchBytes / bytesPerPixel);
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { w, h, 1 };

    vkCmdCopyImageToBuffer(
        cmd,
        _depthImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        stagingBuffer,
        1,
        &region);

    VkImageMemoryBarrier backToDepth{};
    backToDepth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    backToDepth.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    backToDepth.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    backToDepth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToDepth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToDepth.image = _depthImage;
    backToDepth.subresourceRange = toTransfer.subresourceRange;
    backToDepth.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    backToDepth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &backToDepth);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphicsQueue);

    void* mapped = nullptr;
    if (vkMapMemory(_device, stagingMemory, 0, bufferSize, 0, &mapped) != VK_SUCCESS) {
        vkFreeCommandBuffers(_device, _commandPool, 1, &cmd);
        vkDestroyBuffer(_device, stagingBuffer, nullptr);
        vkFreeMemory(_device, stagingMemory, nullptr);
        return false;
    }

    const uint8_t* src = static_cast<const uint8_t*>(mapped);
    outDepth.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const size_t si =
                static_cast<size_t>(y) * static_cast<size_t>(rowPitchBytes) + static_cast<size_t>(x) * 4;
            uint32_t packed = 0;
            std::memcpy(&packed, src + si, sizeof(uint32_t));
            float d01 = 0.0f;
            if (depthFmt == VK_FORMAT_D32_SFLOAT || depthFmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
                d01 = *reinterpret_cast<const float*>(&packed);
            }
            else if (depthFmt == VK_FORMAT_D24_UNORM_S8_UINT) {
                d01 = static_cast<float>(packed & 0x00FFFFFFu) / 16777215.0f;
            }
            else {
                vkUnmapMemory(_device, stagingMemory);
                vkFreeCommandBuffers(_device, _commandPool, 1, &cmd);
                vkDestroyBuffer(_device, stagingBuffer, nullptr);
                vkFreeMemory(_device, stagingMemory, nullptr);
                return false;
            }
            outDepth[static_cast<size_t>(y) * w + x] = std::clamp(d01, 0.0f, 1.0f);
        }
    }

    vkUnmapMemory(_device, stagingMemory);
    vkFreeCommandBuffers(_device, _commandPool, 1, &cmd);
    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingMemory, nullptr);

    outW = w;
    outH = h;
    return true;
}

bool VulkanWindow::readDepthAtCursor(double mouseX, double mouseY, float& outDepth01) {
    int windowWidth = 0, windowHeight = 0;
    int renderW = 0, renderH = 0;
    getQtWindowSize(windowWidth, windowHeight);
    getRenderTargetPixelSize(renderW, renderH);
    if (windowWidth <= 0 || windowHeight <= 0 || renderW <= 0 || renderH <= 0) {
        return false;
    }

    // Window-space cursor → same pixel grid as the swap chain / viewport (HiDPI-safe).
    const double framebufferX = mouseX * static_cast<double>(renderW) / static_cast<double>(windowWidth);
    const double framebufferY = mouseY * static_cast<double>(renderH) / static_cast<double>(windowHeight);

    const uint32_t pixelX = static_cast<uint32_t>(std::clamp<int>(
        static_cast<int>(framebufferX), 0, renderW - 1));
    // Vulkan attachment (0,0) is upper-left; row index matches Qt top-left mouse Y (no GL-style flip).
    const uint32_t pixelY = static_cast<uint32_t>(std::clamp<int>(
        static_cast<int>(framebufferY), 0, renderH - 1));

    return readDepthAtFramebufferPixel(pixelX, pixelY, outDepth01);
}

glm::vec3 VulkanWindow::screenToWorldByDepth(double mouseX, double mouseY, float depth01) const {
    int windowWidth = 0, windowHeight = 0;
    int framebufferWidth = 0, framebufferHeight = 0;
    getQtWindowSize(windowWidth, windowHeight);
    getQtFramebufferSize(framebufferWidth, framebufferHeight);
    if (windowWidth <= 0 || windowHeight <= 0 || framebufferWidth <= 0 || framebufferHeight <= 0) {
        return glm::vec3(0.0f);
    }

    // Cursor position is in window coordinates; convert to framebuffer pixels.
    const double framebufferX = mouseX * static_cast<double>(framebufferWidth) / static_cast<double>(windowWidth);
    const double framebufferY = mouseY * static_cast<double>(framebufferHeight) / static_cast<double>(windowHeight);

    const float ndcX = static_cast<float>((2.0 * framebufferX) / static_cast<double>(framebufferWidth) - 1.0);
    // Vulkan viewport coordinates have top-left origin with positive viewport height.
    const float ndcY = static_cast<float>((2.0 * framebufferY) / static_cast<double>(framebufferHeight) - 1.0);
    const float clampedDepth = std::clamp(depth01, 0.0f, 1.0f);
    if (clampedDepth >= 0.999f) {
        return glm::vec3(0.0f);
    }
    const float ndcZ = clampedDepth; // Vulkan NDC z range is [0, 1]

    // Unproject from NDC back to world using inverse view-projection.
    const glm::mat4 invVP = glm::inverse(buildProjMatrix() * buildViewMatrix());
    const glm::vec4 worldH = invVP * glm::vec4(ndcX, ndcY, ndcZ, 1.0f);
    if (!std::isfinite(worldH.w) || std::fabs(worldH.w) < 1e-7f) {
        return glm::vec3(0.0f);
    }
    const glm::vec3 world = glm::vec3(worldH) / worldH.w;
    if (!std::isfinite(world.x) || !std::isfinite(world.y) || !std::isfinite(world.z)) {
        return glm::vec3(0.0f);
    }
    return world;
}

glm::vec3 VulkanWindow::worldToScreen(const glm::vec3& world) const {
    int windowWidth = 0, windowHeight = 0;
    int framebufferWidth = 0, framebufferHeight = 0;
    getQtWindowSize(windowWidth, windowHeight);
    getQtFramebufferSize(framebufferWidth, framebufferHeight);
    if (windowWidth <= 0 || windowHeight <= 0 || framebufferWidth <= 0 || framebufferHeight <= 0) {
        return glm::vec3(0.0f);
    }

    // Project world -> clip -> NDC -> window-space cursor coordinates.
    const glm::vec4 clip = buildProjMatrix() * buildViewMatrix() * glm::vec4(world, 1.0f);
    if (clip.w == 0.0f) {
        return glm::vec3(0.0f);
    }

    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    const double framebufferX = (static_cast<double>(ndc.x) + 1.0) * 0.5 * static_cast<double>(framebufferWidth);
    const double framebufferY = (static_cast<double>(ndc.y) + 1.0) * 0.5 * static_cast<double>(framebufferHeight);

    const float screenX = static_cast<float>(framebufferX * static_cast<double>(windowWidth) / static_cast<double>(framebufferWidth));
    const float screenY = static_cast<float>(framebufferY * static_cast<double>(windowHeight) / static_cast<double>(framebufferHeight));
    const float depth01 = std::clamp(ndc.z, 0.0f, 1.0f);

    return glm::vec3(screenX, screenY, depth01);
}

void VulkanWindow::cursorPosCallbackImpl(double xpos, double ypos) {
    if (xpos != _lastX || ypos != _lastY) {
        _overlayDirty = true;
    }
    const Qt::KeyboardModifiers km = QGuiApplication::keyboardModifiers();
    const bool shiftHeld = km.testFlag(Qt::ShiftModifier);
    const bool ctrlHeld = km.testFlag(Qt::ControlModifier);
    const bool altHeld = km.testFlag(Qt::AltModifier) || km.testFlag(Qt::MetaModifier);

    // If neither left mouse nor modifier keys are active, just update last pos and return
    if (!_leftMousePressed && !shiftHeld && !ctrlHeld && !altHeld) {
        _lastX = xpos;
        _lastY = ypos;
        return;
    }

    // If a modifier was just pressed, initialize last positions to avoid a large jump
    if (shiftHeld && !_shiftPressedPrev && !_leftMousePressed) {
        _lastX = xpos; _lastY = ypos;
    }
    if (ctrlHeld && !_ctrlPressedPrev && !_leftMousePressed) {
        _lastX = xpos; _lastY = ypos;
    }
    if (altHeld && !_altPressedPrev && !_leftMousePressed) {
        _lastX = xpos; _lastY = ypos;
    }

    double deltaX = xpos - _lastX;
    double deltaY = ypos - _lastY;

    // Controls:
    // - Shift: pan view
    // - Ctrl: orbit (change rotation angles)
    // - Alt: move only the coordinate axis origin in camera X/Y directions
    if (shiftHeld) { // Pan (translate target) when shift is held
        float transX = deltaX * (_ortho[1] - _ortho[0]) / _viewport[0];
        float transY = -deltaY * (_ortho[3] - _ortho[2]) / _viewport[1];

        _ortho[0] -= transX;
        _ortho[1] -= transX;
        _ortho[2] -= transY;
        _ortho[3] -= transY;
        _overlayDirty = true;
    }
    else if (ctrlHeld) { // Rotate when ctrl is held          
        float rotateY = deltaX * 0.8f;
        float rotateX = deltaY * 0.8f;

        setRotation(_rotation + glm::vec2(rotateX, rotateY));
    }
    else if (altHeld) { // Alt + mouse move: translate coordinate (axes) position
        // camera x, y
        float transX = deltaX * (_ortho[1] - _ortho[0]) / _viewport[0];
        float transY = -deltaY * (_ortho[3] - _ortho[2]) / _viewport[1];

        glm::quat quatX = glm::angleAxis(glm::radians(-_rotation[0]), glm::vec3(1, 0, 0));
        glm::quat quatY = glm::angleAxis(glm::radians(-_rotation[1]), glm::vec3(0, 1, 0));

        // camera x, y world vector
        glm::vec3 viewX = quatY * quatX * glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 viewY = quatY * quatX * glm::vec3(0.0f, 1.0f, 0.0f);

        _coordinate += transX * viewX;
        _coordinate += transY * viewY;

        _ortho[0] -= transX;
        _ortho[1] -= transX;
        _ortho[2] -= transY;
        _ortho[3] -= transY;
        _overlayDirty = true;
    }

    _lastX = xpos;
    _lastY = ypos;

    // store previous modifier state for next callback to avoid jumps
    _shiftPressedPrev = shiftHeld;
    _ctrlPressedPrev = ctrlHeld;
    _altPressedPrev = altHeld;
}

void VulkanWindow::mouseButtonCallbackImpl(Qt::MouseButton button, QEvent::Type type, Qt::KeyboardModifiers mods, double localX, double localY, long long timestampMs) {
    _lastX = localX;
    _lastY = localY;
    if (button != Qt::LeftButton) {
        return;
    }
    if (type == QEvent::MouseButtonPress) {
        _leftMousePressed = true;
        const double x = localX;
        const double y = localY;

        const bool modPick = mods.testFlag(Qt::AltModifier) || mods.testFlag(Qt::ControlModifier) || mods.testFlag(Qt::MetaModifier);
        if (_lastLeftClickTimestampMs != 0 && (timestampMs - _lastLeftClickTimestampMs) < 400LL && modPick) {
            float depth01 = 1.0f;
            if (readDepthAtCursor(x, y, depth01)) {
                const glm::vec3 pickedCoordinate = screenToWorldByDepth(x, y, depth01);
                std::cout << std::fixed << std::setprecision(4)
                    << "Alt+DoubleClick x=" << x
                    << " y=" << y
                    << " zDepth=" << depth01
                    << " -> coordinateNew=(" << pickedCoordinate.x << ", " << pickedCoordinate.y << ", " << pickedCoordinate.z << ")\n";

                glm::vec3 coordinateScreen = worldToScreen(_coordinate);

                double deltaX = x - coordinateScreen[0];
                double deltaY = y - coordinateScreen[1];

                float transX = deltaX * (_ortho[1] - _ortho[0]) / _viewport[0];
                float transY = -deltaY * (_ortho[3] - _ortho[2]) / _viewport[1];

                _ortho[0] -= transX;
                _ortho[1] -= transX;
                _ortho[2] -= transY;
                _ortho[3] -= transY;

                _coordinate = pickedCoordinate;
                _overlayDirty = true;
            }
            else {
                std::cout << "Alt+DoubleClick depth read failed at x=" << x << " y=" << y << "\n";
            }
            _lastLeftClickTimestampMs = 0;
        }
        else {
            _lastLeftClickTimestampMs = timestampMs;
        }
    }
    else if (type == QEvent::MouseButtonRelease) {
        _leftMousePressed = false;
    }
}

void VulkanWindow::scrollCallbackImpl(float yoffset) {
    const Qt::KeyboardModifiers km = QGuiApplication::keyboardModifiers();
    const bool altHeld = km.testFlag(Qt::AltModifier) || km.testFlag(Qt::MetaModifier);
    if (altHeld) {
        float transZ = static_cast<float>(yoffset) * (_ortho[3] - _ortho[2]) / _viewport[1] * 20;

        glm::quat quatX = glm::angleAxis(glm::radians(-_rotation[0]), glm::vec3(1, 0, 0));
        glm::quat quatY = glm::angleAxis(glm::radians(-_rotation[1]), glm::vec3(0, 1, 0));

        glm::vec3 viewZ = quatY * quatX * glm::vec3(0.0f, 0.0f, 1.0f);

        _coordinate += transZ * viewZ;
        _overlayDirty = true;
        return;
    }

    float scale = 1.0f - yoffset * 0.1f;
    _ortho *= scale; // _coorinate(0, 0)
    _overlayDirty = true;
}

void VulkanWindow::initVulkan() {
    if (_ownsVkInstance) {
        createInstance();
    }
    if (_instance == VK_NULL_HANDLE) {
        throw std::runtime_error("VkInstance is null.");
    }
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createRulerPipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    // No frame has been rendered yet; depth readback here crashes some drivers / mismatches layout.
    createVertexBuffer(false);
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

void VulkanWindow::createInstance() {
    gUseValidation = ENABLE_VALIDATION && checkValidationLayerSupport();
    if (ENABLE_VALIDATION && !gUseValidation) {
        std::cerr << "Warning: validation layers not available; continuing without validation.\n";
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "CoordAxis";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    auto extensions = getRequiredExtensions();

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT dbgCI{};
    if (gUseValidation) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        fillDebugMessengerCI(dbgCI);
        ci.pNext = &dbgCI;
    }

    if (vkCreateInstance(&ci, nullptr, &_instance) != VK_SUCCESS) {
        if (gUseValidation) {
            // Retry once without validation/debug utils for environments
            // where these components are not installed.
            gUseValidation = false;
            auto fallbackExtensions = getRequiredExtensions();
            ci.enabledExtensionCount = static_cast<uint32_t>(fallbackExtensions.size());
            ci.ppEnabledExtensionNames = fallbackExtensions.data();
            ci.enabledLayerCount = 0;
            ci.ppEnabledLayerNames = nullptr;
            ci.pNext = nullptr;
            if (vkCreateInstance(&ci, nullptr, &_instance) == VK_SUCCESS) {
                return;
            }
        }
        throw std::runtime_error("vkCreateInstance failed.");
    }
}

bool VulkanWindow::checkValidationLayerSupport() {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());
    for (const char* name : VALIDATION_LAYERS) {
        bool found = false;
        for (auto& p : available)
            if (strcmp(name, p.layerName) == 0) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

std::vector<const char*> VulkanWindow::getRequiredExtensions() {
    std::vector<const char*> exts{ VK_KHR_SURFACE_EXTENSION_NAME };
#if defined(Q_OS_WIN)
    exts.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(Q_OS_LINUX)
    exts.push_back("VK_KHR_xcb_surface");
#endif
    if (gUseValidation) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return exts;
}

void VulkanWindow::fillDebugMessengerCI(VkDebugUtilsMessengerCreateInfoEXT& ci) {
    ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
}

void VulkanWindow::setupDebugMessenger() {
    if (!gUseValidation) return;
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    fillDebugMessengerCI(ci);
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn || fn(_instance, &ci, nullptr, &_debugMessenger) != VK_SUCCESS) {
        std::cerr << "Warning: debug messenger unavailable; continuing without it.\n";
        _debugMessenger = VK_NULL_HANDLE;
        gUseValidation = false;
    }
}

void VulkanWindow::createSurface() {
    if (!_qtWindow || !_qVulkanInstance) {
        throw std::runtime_error("Qt window / QVulkanInstance not set before createSurface().");
    }
    _surface = QVulkanInstance::surfaceForWindow(_qtWindow);
    if (_surface == VK_NULL_HANDLE) {
        throw std::runtime_error("QVulkanInstance::surfaceForWindow failed.");
    }
}

void VulkanWindow::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(_instance, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan-capable GPU found.");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(_instance, &count, devs.data());
    for (auto& d : devs)
        if (isDeviceSuitable(d)) { _physicalDevice = d; break; }
    if (_physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("No suitable GPU found.");
    VkPhysicalDeviceProperties physProps{};
    vkGetPhysicalDeviceProperties(_physicalDevice, &physProps);
    _copyRowPitchAlign = physProps.limits.optimalBufferCopyRowPitchAlignment;
    if (_copyRowPitchAlign < 1) {
        _copyRowPitchAlign = 1;
    }
    _copyOffsetAlign = physProps.limits.optimalBufferCopyOffsetAlignment;
    if (_copyOffsetAlign < 1) {
        _copyOffsetAlign = 1;
    }
}

bool VulkanWindow::isDeviceSuitable(VkPhysicalDevice dev) {
    auto idx = findQueueFamilies(dev);
    bool extOk = checkDeviceExtensionSupport(dev);
    bool swapOk = false;
    if (extOk) {
        auto sc = querySwapChainSupport(dev);
        swapOk = !sc.formats.empty() && !sc.presentModes.empty();
    }
    return idx.isComplete() && extOk && swapOk;
}

bool VulkanWindow::checkDeviceExtensionSupport(VkPhysicalDevice dev) {
    uint32_t cnt;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &cnt, nullptr);
    std::vector<VkExtensionProperties> available(cnt);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &cnt, available.data());
    std::set<std::string> required(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
    for (auto& e : available) required.erase(e.extensionName);
    return required.empty();
}

QueueFamilyIndices VulkanWindow::findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices idx;
    uint32_t cnt;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &cnt, nullptr);
    std::vector<VkQueueFamilyProperties> fams(cnt);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &cnt, fams.data());
    for (uint32_t i = 0; i < cnt; i++) {
        if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            idx.graphicsFamily = i;
        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, _surface, &present);
        if (present) idx.presentFamily = i;
        if (idx.isComplete()) break;
    }
    return idx;
}

SwapChainSupportDetails VulkanWindow::querySwapChainSupport(VkPhysicalDevice dev) {
    SwapChainSupportDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, _surface, &d.capabilities);
    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, _surface, &n, nullptr);
    d.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, _surface, &n, d.formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, _surface, &n, nullptr);
    d.presentModes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, _surface, &n, d.presentModes.data());
    return d;
}

void VulkanWindow::createLogicalDevice() {
    auto idx = findQueueFamilies(_physicalDevice);
    std::set<uint32_t> uniqueFams = {
        idx.graphicsFamily.value(), idx.presentFamily.value()
    };

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    for (uint32_t fam : uniqueFams) {
        std::set<uint32_t> uniqueFams = {
            idx.graphicsFamily.value(), idx.presentFamily.value()
        };
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = fam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueCIs.push_back(qi);
    }
    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = (uint32_t)queueCIs.size();
    ci.pQueueCreateInfos = queueCIs.data();
    ci.pEnabledFeatures = &features;
    ci.enabledExtensionCount = (uint32_t)DEVICE_EXTENSIONS.size();
    ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    if (ENABLE_VALIDATION) {
        ci.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
        ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }
    if (vkCreateDevice(_physicalDevice, &ci, nullptr, &_device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed.");

    vkGetDeviceQueue(_device, idx.graphicsFamily.value(), 0, &_graphicsQueue);
    vkGetDeviceQueue(_device, idx.presentFamily.value(), 0, &_presentQueue);
}

VkExtent2D VulkanWindow::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return caps.currentExtent;
    }
    int w = 0;
    int h = 0;
    getQtFramebufferSize(w, h);
    return {
        std::clamp(static_cast<uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(static_cast<uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height)
    };
}

VkSurfaceFormatKHR VulkanWindow::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& fmts) {
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return fmts[0];
}

VkPresentModeKHR VulkanWindow::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto& m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            return m;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

void VulkanWindow::createSwapChain() {
    auto sc = querySwapChainSupport(_physicalDevice);
    auto fmt = chooseSurfaceFormat(sc.formats);
    auto pm = choosePresentMode(sc.presentModes);
    auto ext = chooseSwapExtent(sc.capabilities);

    uint32_t imgCount = sc.capabilities.minImageCount + 1;
    if (sc.capabilities.maxImageCount > 0)
        imgCount = std::min(imgCount, sc.capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = _surface;
    ci.minImageCount = imgCount;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto idx = findQueueFamilies(_physicalDevice);
    uint32_t queueFamilyIndices[] = {
        idx.graphicsFamily.value(), idx.presentFamily.value()
    };
    if (idx.graphicsFamily != idx.presentFamily) {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = sc.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(_device, &ci, nullptr, &_swapChain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed.");

    vkGetSwapchainImagesKHR(_device, _swapChain, &imgCount, nullptr);
    _swapChainImages.resize(imgCount);
    vkGetSwapchainImagesKHR(_device, _swapChain, &imgCount, _swapChainImages.data());

    _swapChainImageFormat = fmt.format;
    _swapChainExtent = ext;
}

void VulkanWindow::createImageViews() {
    _swapChainImageViews.resize(_swapChainImages.size());
    for (size_t i = 0; i < _swapChainImages.size(); i++)
        _swapChainImageViews[i] = createImageView(
            _swapChainImages[i], _swapChainImageFormat,
            VK_IMAGE_ASPECT_COLOR_BIT);
}

VkImageView VulkanWindow::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = format;
    ci.subresourceRange.aspectMask = aspectFlags;
    ci.subresourceRange.baseMipLevel = 0;
    ci.subresourceRange.levelCount = 1;
    ci.subresourceRange.baseArrayLayer = 0;
    ci.subresourceRange.layerCount = 1;
    VkImageView view;
    if (vkCreateImageView(_device, &ci, nullptr, &view) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateImageView failed.");
    }
    return view;
}

void VulkanWindow::createRenderPass() {
    // color attachment
    VkAttachmentDescription colorAtt{};
    colorAtt.format = _swapChainImageFormat;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // depth attachment
    VkAttachmentDescription depthAtt{};
    depthAtt.format = findDepthFormat();
    depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Depth is read back for picking, so it must be preserved after render pass.
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = { colorAtt, depthAtt };
    VkRenderPassCreateInfo rpCI{};
    rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = (uint32_t)attachments.size();
    rpCI.pAttachments = attachments.data();
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies = &dep;

    if (vkCreateRenderPass(_device, &rpCI, nullptr, &_renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed.");
}

void VulkanWindow::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 1;
    ci.pBindings = &uboBinding;

    if (vkCreateDescriptorSetLayout(_device, &ci, nullptr, &_descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout failed.");
}

void VulkanWindow::createGraphicsPipeline() {
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;
#ifdef TEST_USE_PREBUILT_SPV_ONLY
    auto vertCode = readSPVFile("shaders/shader.vert.spv");
    auto fragCode = readSPVFile("shaders/shader.frag.spv");
    vertModule = createShaderModule(_device, vertCode);
    fragModule = createShaderModule(_device, fragCode);
#else
    try {
        auto vertCode = readSPVFile("shaders/shader.vert.spv");
        auto fragCode = readSPVFile("shaders/shader.frag.spv");
        vertModule = createShaderModule(_device, vertCode);
        fragModule = createShaderModule(_device, fragCode);
    }
    catch (...) {
        auto vertCode = compileGLSL(readTextFile("shaders/shader.vert"), shaderc_glsl_vertex_shader, "shader.vert");
        auto fragCode = compileGLSL(readTextFile("shaders/shader.frag"), shaderc_glsl_fragment_shader, "shader.frag");
        vertModule = createShaderModule(_device, vertCode);
        fragModule = createShaderModule(_device, fragCode);
    }
#endif

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    //  Vertex input 
    auto bindDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputCI{};
    vertexInputCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputCI.vertexBindingDescriptionCount = 1;
    vertexInputCI.pVertexBindingDescriptions = &bindDesc;
    vertexInputCI.vertexAttributeDescriptionCount = (uint32_t)attrDesc.size();
    vertexInputCI.pVertexAttributeDescriptions = attrDesc.data();

    //  Input assembly   LINE_LIST for axes 
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
    inputAssemblyCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    //  Viewport / scissor (dynamic) 
    VkPipelineViewportStateCreateInfo viewportCI{};
    viewportCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportCI.viewportCount = 1;
    viewportCI.scissorCount = 1;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicCI{};
    dynamicCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicCI.dynamicStateCount = (uint32_t)dynamicStates.size();
    dynamicCI.pDynamicStates = dynamicStates.data();

    //  Rasteriser 
    VkPipelineRasterizationStateCreateInfo rasterCI{};
    rasterCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterCI.polygonMode = VK_POLYGON_MODE_FILL;
    rasterCI.lineWidth = 1.0f;          // thick axes
    rasterCI.cullMode = VK_CULL_MODE_NONE;
    rasterCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    //  Multisampling 
    VkPipelineMultisampleStateCreateInfo msCI{};
    msCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    //  Depth / stencil  (depth test ON) 
    VkPipelineDepthStencilStateCreateInfo depthCI{};
    depthCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthCI.depthTestEnable = VK_TRUE;
    depthCI.depthWriteEnable = VK_TRUE;
    depthCI.depthCompareOp = VK_COMPARE_OP_LESS;
    depthCI.minDepthBounds = 0.0f;
    depthCI.maxDepthBounds = 1.0f;

    //  Color blend 
    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blendCI{};
    blendCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendCI.attachmentCount = 1;
    blendCI.pAttachments = &blendAtt;

    //  Pipeline layout 
    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts = &_descriptorSetLayout;

    if (vkCreatePipelineLayout(_device, &layoutCI, nullptr, &_pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout failed.");

    // Create triangle pipeline (for cube)
    VkGraphicsPipelineCreateInfo triCI{};
    triCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    triCI.stageCount = 2;
    triCI.pStages = stages;
    triCI.pVertexInputState = &vertexInputCI;
    // triangles
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    triCI.pInputAssemblyState = &inputAssemblyCI;
    triCI.pViewportState = &viewportCI;
    triCI.pRasterizationState = &rasterCI;
    triCI.pMultisampleState = &msCI;
    triCI.pDepthStencilState = &depthCI;
    triCI.pColorBlendState = &blendCI;
    triCI.pDynamicState = &dynamicCI;
    triCI.layout = _pipelineLayout;
    triCI.renderPass = _renderPass;
    triCI.subpass = 0;

    if (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1,
        &triCI, nullptr,
        &_trianglePipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (triangle) failed.");

    // Create line pipeline (for axes) - set input assembly back to lines
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    VkPipelineRasterizationStateCreateInfo lineRaster = rasterCI;
    lineRaster.lineWidth = 2.0f;

    VkGraphicsPipelineCreateInfo lineCI = triCI;
    lineCI.pInputAssemblyState = &inputAssemblyCI;
    lineCI.pRasterizationState = &lineRaster;
    // Draw lines as an overlay on triangle surfaces.
    VkPipelineDepthStencilStateCreateInfo lineDepthCI = depthCI;
    lineDepthCI.depthTestEnable = VK_TRUE;
    lineDepthCI.depthWriteEnable = VK_FALSE;
    lineDepthCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    lineCI.pDepthStencilState = &lineDepthCI;

    if (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1,
        &lineCI, nullptr,
        &_linePipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (line) failed.");

    vkDestroyShaderModule(_device, vertModule, nullptr);
    vkDestroyShaderModule(_device, fragModule, nullptr);
}

void VulkanWindow::createRulerPipeline() {
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;
#ifdef TEST_USE_PREBUILT_SPV_ONLY
    auto vertCode = readSPVFile("shaders/ruler.vert.spv");
    auto fragCode = readSPVFile("shaders/ruler.frag.spv");
    vertModule = createShaderModule(_device, vertCode);
    fragModule = createShaderModule(_device, fragCode);
#else
    try {
        auto vertCode = readSPVFile("shaders/ruler.vert.spv");
        auto fragCode = readSPVFile("shaders/ruler.frag.spv");
        vertModule = createShaderModule(_device, vertCode);
        fragModule = createShaderModule(_device, fragCode);
    }
    catch (...) {
        auto vertCode = compileGLSL(readTextFile("shaders/ruler.vert"), shaderc_glsl_vertex_shader, "ruler.vert");
        auto fragCode = compileGLSL(readTextFile("shaders/ruler.frag"), shaderc_glsl_fragment_shader, "ruler.frag");
        vertModule = createShaderModule(_device, vertCode);
        fragModule = createShaderModule(_device, fragCode);
    }
#endif

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    auto bindDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputCI{};
    vertexInputCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputCI.vertexBindingDescriptionCount = 1;
    vertexInputCI.pVertexBindingDescriptions = &bindDesc;
    vertexInputCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertexInputCI.pVertexAttributeDescriptions = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
    inputAssemblyCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineViewportStateCreateInfo viewportCI{};
    viewportCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportCI.viewportCount = 1;
    viewportCI.scissorCount = 1;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicCI{};
    dynamicCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicCI.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicCI.pDynamicStates = dynamicStates.data();

    VkPipelineRasterizationStateCreateInfo rasterCI{};
    rasterCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterCI.polygonMode = VK_POLYGON_MODE_FILL;
    rasterCI.lineWidth = 2.0f;
    rasterCI.cullMode = VK_CULL_MODE_NONE;
    rasterCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo msCI{};
    msCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthCI{};
    depthCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthCI.depthTestEnable = VK_FALSE;
    depthCI.depthWriteEnable = VK_FALSE;
    depthCI.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blendCI{};
    blendCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendCI.attachmentCount = 1;
    blendCI.pAttachments = &blendAtt;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vertexInputCI;
    ci.pInputAssemblyState = &inputAssemblyCI;
    ci.pViewportState = &viewportCI;
    ci.pRasterizationState = &rasterCI;
    ci.pMultisampleState = &msCI;
    ci.pDepthStencilState = &depthCI;
    ci.pColorBlendState = &blendCI;
    ci.pDynamicState = &dynamicCI;
    ci.layout = _pipelineLayout;
    ci.renderPass = _renderPass;
    ci.subpass = 0;

    if (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &ci, nullptr, &_rulerPipeline) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateGraphicsPipelines (ruler) failed.");
    }

    vkDestroyShaderModule(_device, vertModule, nullptr);
    vkDestroyShaderModule(_device, fragModule, nullptr);
}

void VulkanWindow::createCommandPool() {
    auto idx = findQueueFamilies(_physicalDevice);
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = idx.graphicsFamily.value();
    if (vkCreateCommandPool(_device, &ci, nullptr, &_commandPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateCommandPool failed.");
}

VkFormat VulkanWindow::findDepthFormat() {
    return findSupportedFormat(
        { VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT);
}

VkFormat VulkanWindow::findSupportedFormat(const std::vector<VkFormat>& candidates,
VkImageTiling tiling,
VkFormatFeatureFlags features) {
    for (VkFormat fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(_physicalDevice, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features) {
            return fmt;
        }
        if (tiling == VK_IMAGE_TILING_OPTIMAL &&
            (props.optimalTilingFeatures & features) == features) {
            return fmt;
        }
    }
    throw std::runtime_error("Failed to find supported format.");
}

uint32_t VulkanWindow::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("Failed to find suitable memory type.");
}

void VulkanWindow::createImage(uint32_t w, uint32_t h, VkFormat fmt,
VkImageTiling tiling, VkImageUsageFlags usage,
VkMemoryPropertyFlags props,
VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = fmt;
    ci.tiling = tiling;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = usage;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(_device, &ci, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateImage failed.");
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(_device, image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, props);

    if (vkAllocateMemory(_device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("vkAllocateMemory (image) failed.");
    }

    vkBindImageMemory(_device, image, memory, 0);
}

void VulkanWindow::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();
    createImage(
        _swapChainExtent.width, _swapChainExtent.height,
        depthFormat, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        _depthImage, _depthImageMemory);
    _depthImageView = createImageView(
        _depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void VulkanWindow::createFramebuffers() {
    _swapChainFramebuffers.resize(_swapChainImageViews.size());
    for (size_t i = 0; i < _swapChainImageViews.size(); i++) {
        std::array<VkImageView, 2> attachments = {
            _swapChainImageViews[i], _depthImageView
        };
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = _renderPass;
        ci.attachmentCount = (uint32_t)attachments.size();
        ci.pAttachments = attachments.data();
        ci.width = _swapChainExtent.width;
        ci.height = _swapChainExtent.height;
        ci.layers = 1;
        if (vkCreateFramebuffer(_device, &ci, nullptr,
            &_swapChainFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer failed.");
    }
}

void VulkanWindow::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = _commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(_device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{ 0, 0, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &copyRegion);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphicsQueue);
    vkFreeCommandBuffers(_device, _commandPool, 1, &cmd);
}

void VulkanWindow::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
VkMemoryPropertyFlags props,
VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(_device, &ci, nullptr, &buf) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateBuffer failed.");
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(_device, buf, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, props);
    if (vkAllocateMemory(_device, &allocInfo, nullptr, &mem) != VK_SUCCESS) {
        throw std::runtime_error("vkAllocateMemory failed.");
    }
    vkBindBufferMemory(_device, buf, mem, 0);
}

void VulkanWindow::createVertexBuffer(bool probeDepthForRuler) {
    std::vector<Vertex> axisVertices = AXIS_VERTICES;
    std::vector<Vertex> rulerVertices;
    float depth01 = 1.0f;
    bool hasDepthWorld = false;
    glm::vec3 depthWorld(0.0f, 0.0f, 0.0f);
    if (probeDepthForRuler) {
        hasDepthWorld = readDepthAtCursor(_lastX, _lastY, depth01);
        if (hasDepthWorld) {
            depthWorld = screenToWorldByDepth(_lastX, _lastY, depth01);
            if (!std::isfinite(depthWorld.x) || !std::isfinite(depthWorld.y) || !std::isfinite(depthWorld.z)) {
                hasDepthWorld = false;
                depthWorld = glm::vec3(0.0f, 0.0f, 0.0f);
            }
        }
    }
    appendVulkanRulerVertices(
        rulerVertices,
        buildRulerOrtho(),
        static_cast<float>(_swapChainExtent.width),
        static_cast<float>(_swapChainExtent.height),
        static_cast<float>(_lastX),
        static_cast<float>(_lastY),
        hasDepthWorld,
        depthWorld);
    _axisVertexCount = static_cast<uint32_t>(axisVertices.size());
    _rulerVertexCount = static_cast<uint32_t>(rulerVertices.size());
    std::vector<Vertex> applicationFaceVertices = _applicationVertices;
    _applicationTriangleVertexCount = static_cast<uint32_t>(applicationFaceVertices.size());

    std::vector<Vertex> applicationLineVertices;
    applicationLineVertices.reserve((_applicationTriangleVertexCount / 3) * 6);
    for (uint32_t i = 0; i + 2 < _applicationTriangleVertexCount; i += 3) {
        Vertex v0 = applicationFaceVertices[i];
        Vertex v1 = applicationFaceVertices[i + 1];
        Vertex v2 = applicationFaceVertices[i + 2];
        applicationLineVertices.push_back(v0);
        applicationLineVertices.push_back(v1);
        applicationLineVertices.push_back(v1);
        applicationLineVertices.push_back(v2);
        applicationLineVertices.push_back(v2);
        applicationLineVertices.push_back(v0);
    }
    _applicationLineVertexCount = static_cast<uint32_t>(applicationLineVertices.size());

    const size_t totalVertexCount =
        static_cast<size_t>(_axisVertexCount) +
        static_cast<size_t>(_rulerVertexCount) +
        static_cast<size_t>(_applicationTriangleVertexCount) +
        static_cast<size_t>(_applicationLineVertexCount);
    VkDeviceSize size = sizeof(Vertex) * totalVertexCount;

    VkBuffer       stagingBuf;
    VkDeviceMemory stagingMem;
    createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuf, stagingMem);

    void* data;
    vkMapMemory(_device, stagingMem, 0, size, 0, &data);
    memcpy(data, axisVertices.data(), sizeof(Vertex) * _axisVertexCount);
    if (_rulerVertexCount > 0) {
        memcpy(static_cast<char*>(data) + sizeof(Vertex) * _axisVertexCount,
            rulerVertices.data(),
            sizeof(Vertex) * _rulerVertexCount);
    }
    if (_applicationTriangleVertexCount > 0) {
        memcpy(static_cast<char*>(data) + sizeof(Vertex) * (_axisVertexCount + _rulerVertexCount),
            applicationFaceVertices.data(),
            sizeof(Vertex) * _applicationTriangleVertexCount);
    }
    if (_applicationLineVertexCount > 0) {
        memcpy(static_cast<char*>(data) + sizeof(Vertex) * (_axisVertexCount + _rulerVertexCount + _applicationTriangleVertexCount),
            applicationLineVertices.data(),
            sizeof(Vertex) * _applicationLineVertexCount);
    }
    vkUnmapMemory(_device, stagingMem);

    createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        _vertexBuffer, _vertexBufferMemory);

    copyBuffer(stagingBuf, _vertexBuffer, size);
    vkDestroyBuffer(_device, stagingBuf, nullptr);
    vkFreeMemory(_device, stagingMem, nullptr);
}

void VulkanWindow::createUniformBuffers() {
    VkDeviceSize bufSize = sizeof(UniformBufferObject);
    // Per frame: scene UBO, axis-gizmo UBO, ruler-overlay UBO.
    size_t totalUBOs = MAX_FRAMES_IN_FLIGHT * 3;
    _uniformBuffers.resize(totalUBOs);
    _uniformBuffersMemory.resize(totalUBOs);
    _uniformBuffersMapped.resize(totalUBOs);

    for (size_t i = 0; i < totalUBOs; i++) {
        createBuffer(bufSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            _uniformBuffers[i], _uniformBuffersMemory[i]);
        vkMapMemory(_device, _uniformBuffersMemory[i], 0, bufSize, 0,
            &_uniformBuffersMapped[i]);
    }
}

void VulkanWindow::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Three uniform descriptors are used per frame.
    poolSize.descriptorCount = (uint32_t)MAX_FRAMES_IN_FLIGHT * 3;

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &poolSize;
    ci.maxSets = (uint32_t)MAX_FRAMES_IN_FLIGHT * 3;

    if (vkCreateDescriptorPool(_device, &ci, nullptr, &_descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool failed.");
}

void VulkanWindow::createDescriptorSets() {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = _descriptorPool;
    // Allocate three descriptor sets per frame.
    ai.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT * 3;
    std::vector<VkDescriptorSetLayout> setLayouts(ai.descriptorSetCount, _descriptorSetLayout);
    ai.pSetLayouts = setLayouts.data();

    _descriptorSets.resize(ai.descriptorSetCount);
    if (vkAllocateDescriptorSets(_device, &ai, _descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets failed.");

    // Update descriptor sets for all per-frame UBOs.
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufCube{};
        bufCube.buffer = _uniformBuffers[3 * i];
        bufCube.offset = 0;
        bufCube.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet writeCube{};
        writeCube.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeCube.dstSet = _descriptorSets[3 * i];
        writeCube.dstBinding = 0;
        writeCube.descriptorCount = 1;
        writeCube.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeCube.pBufferInfo = &bufCube;

        VkDescriptorBufferInfo bufAxes{};
        bufAxes.buffer = _uniformBuffers[3 * i + 1];
        bufAxes.offset = 0;
        bufAxes.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet writeAxes{};
        writeAxes.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeAxes.dstSet = _descriptorSets[3 * i + 1];
        writeAxes.dstBinding = 0;
        writeAxes.descriptorCount = 1;
        writeAxes.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeAxes.pBufferInfo = &bufAxes;

        VkDescriptorBufferInfo bufRuler{};
        bufRuler.buffer = _uniformBuffers[3 * i + 2];
        bufRuler.offset = 0;
        bufRuler.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet writeRuler{};
        writeRuler.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeRuler.dstSet = _descriptorSets[3 * i + 2];
        writeRuler.dstBinding = 0;
        writeRuler.descriptorCount = 1;
        writeRuler.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeRuler.pBufferInfo = &bufRuler;

        VkWriteDescriptorSet writes[3] = { writeCube, writeAxes, writeRuler };
        vkUpdateDescriptorSets(_device, 3, writes, 0, nullptr);
    }
}

void VulkanWindow::createCommandBuffers() {
    _commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = _commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)_commandBuffers.size();
    if (vkAllocateCommandBuffers(_device, &ai, _commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers failed.");
}

void VulkanWindow::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("vkBeginCommandBuffer failed.");

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { 0.05f, 0.05f, 0.05f, 1.0f }; // dark bg
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpBI{};
    rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBI.renderPass = _renderPass;
    rpBI.framebuffer = _swapChainFramebuffers[imageIndex];
    rpBI.renderArea.offset = { 0, 0 };
    rpBI.renderArea.extent = _swapChainExtent;
    rpBI.clearValueCount = (uint32_t)clearValues.size();
    rpBI.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.f;
    viewport.y = 0.f;
    viewport.width = (float)_swapChainExtent.width;
    viewport.height = (float)_swapChainExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{ {0, 0}, _swapChainExtent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer     vbs[] = { _vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
    // Bind the scene descriptor set.
    uint32_t cubeDescIndex = _currentFrame * 3;
    uint32_t axesDescIndex = _currentFrame * 3 + 1;
    uint32_t rulerDescIndex = _currentFrame * 3 + 2;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        _pipelineLayout, 0, 1,
        &_descriptorSets[cubeDescIndex], 0, nullptr);

    if (_applicationTriangleVertexCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);
        vkCmdDraw(cmd, _applicationTriangleVertexCount, 1, _axisVertexCount + _rulerVertexCount, 0);
    }

    if (_applicationLineVertexCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _linePipeline);
        vkCmdDraw(cmd, _applicationLineVertexCount, 1, _axisVertexCount + _rulerVertexCount + _applicationTriangleVertexCount, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _linePipeline);
    // Bind axis descriptor set.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        _pipelineLayout, 0, 1,
        &_descriptorSets[axesDescIndex], 0, nullptr);
    vkCmdDraw(cmd, _axisVertexCount, 1, 0, 0);

    if (_rulerVertexCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _rulerPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            _pipelineLayout, 0, 1,
            &_descriptorSets[rulerDescIndex], 0, nullptr);
        vkCmdDraw(cmd, _rulerVertexCount, 1, _axisVertexCount, 0);
    }

    vkCmdEndRenderPass(cmd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("vkEndCommandBuffer failed.");
}

void VulkanWindow::createSyncObjects() {
    _imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    _inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semCI{};
    semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceCI{};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(_device, &semCI, nullptr, &_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(_device, &semCI, nullptr, &_renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(_device, &fenceCI, nullptr, &_inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sync objects.");
    }
}

void VulkanWindow::updateUniformBuffer(uint32_t currentImage) {
    UniformBufferObject ubo{};

    ubo.model = glm::mat4(1.0f);
    ubo.model = glm::translate(ubo.model, _applicationModelTranslation);
    ubo.model = glm::scale(ubo.model, _applicationModelScale);

    glm::quat quatX = glm::angleAxis(glm::radians(-_rotation[0]), glm::vec3(1, 0, 0));
    glm::quat quatY = glm::angleAxis(glm::radians(-_rotation[1]), glm::vec3(0, 1, 0));

    glm::vec3 viewY = quatY * quatX * glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 viewZ = quatY * quatX * glm::vec3(0.0f, 0.0f, -1.0f);

    glm::mat4 view = glm::lookAt(_coordinate - viewZ * 100.0f, _coordinate + viewZ * 100.0f, viewY);
    ubo.view = view;

    glm::mat4 proj = glm::ortho(_ortho[0], _ortho[1], _ortho[3], _ortho[2], 0.1f, 200.0f);
    ubo.proj = proj;

    // write scene UBO into slot 3*currentImage
    void* mappedCube = _uniformBuffersMapped[3 * currentImage];
    memcpy(mappedCube, &ubo, sizeof(ubo));

    UniformBufferObject axesUbo{};
    axesUbo.model = glm::mat4(1.0f);
    axesUbo.model = glm::translate(axesUbo.model, _coordinate);

    float scale = fabs(_ortho[3] - _ortho[2]) * 0.5f;
    axesUbo.model = glm::scale(axesUbo.model, glm::vec3(scale, scale, scale));

    axesUbo.view = view;
    axesUbo.proj = proj;

    void* mappedAxes = _uniformBuffersMapped[3 * currentImage + 1];
    memcpy(mappedAxes, &axesUbo, sizeof(axesUbo));

    UniformBufferObject rulerUbo{};
    rulerUbo.model = glm::mat4(1.0f);
    rulerUbo.view = glm::mat4(1.0f);
    rulerUbo.proj = proj;

    glm::vec4 rulerOrtho = buildRulerOrtho();
    // Same T/B swap as buildProjMatrix() for Vulkan Y convention (_ortho is L,R,B,T in xyzw).
    rulerUbo.proj = glm::ortho(rulerOrtho.x, rulerOrtho.y, rulerOrtho.w, rulerOrtho.z, 0.1f, 200.0f);

    void* mappedRuler = _uniformBuffersMapped[3 * currentImage + 2];
    memcpy(mappedRuler, &rulerUbo, sizeof(rulerUbo));
}

void VulkanWindow::cleanupSwapChain() {
    _depthImageGpuReady = false;
    vkDestroyImageView(_device, _depthImageView, nullptr);
    vkDestroyImage(_device, _depthImage, nullptr);
    vkFreeMemory(_device, _depthImageMemory, nullptr);
    for (auto fb : _swapChainFramebuffers)
        vkDestroyFramebuffer(_device, fb, nullptr);
    for (auto iv : _swapChainImageViews)
        vkDestroyImageView(_device, iv, nullptr);
    vkDestroySwapchainKHR(_device, _swapChain, nullptr);
}

void VulkanWindow::recreateSwapChain() {
    int w = 0;
    int h = 0;
    getQtFramebufferSize(w, h);
    while (w == 0 || h == 0) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
        getQtFramebufferSize(w, h);
    }
    vkDeviceWaitIdle(_device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createDepthResources();
    createFramebuffers();
}

void VulkanWindow::drawFrame() {
    if (_device == VK_NULL_HANDLE || _swapChain == VK_NULL_HANDLE ||
        _swapChainFramebuffers.empty() || _swapChainExtent.width == 0 || _swapChainExtent.height == 0) {
        return;
    }
    syncViewportFromEmbeddedWindow();
    enforceOrthoAspectFromWindow();

    const bool extentChanged =
        (_cachedOverlayExtentWidth != _swapChainExtent.width) ||
        (_cachedOverlayExtentHeight != _swapChainExtent.height);
    const int mouseX = static_cast<int>(_lastX);
    const int mouseY = static_cast<int>(_lastY);
    const bool mouseChanged =
        (mouseX != _cachedMouseOverlayX) || (mouseY != _cachedMouseOverlayY);

    if (_overlayDirty || !_hasCachedOrthoForOverlay || extentChanged || mouseChanged ||
        std::fabs(_cachedOrthoForOverlay.x - _ortho.x) > 1e-5f ||
        std::fabs(_cachedOrthoForOverlay.y - _ortho.y) > 1e-5f ||
        std::fabs(_cachedOrthoForOverlay.z - _ortho.z) > 1e-5f ||
        std::fabs(_cachedOrthoForOverlay.w - _ortho.w) > 1e-5f) {
        rebuildVertexBuffer();
        _cachedOrthoForOverlay = _ortho;
        _hasCachedOrthoForOverlay = true;
        _cachedOverlayExtentWidth = _swapChainExtent.width;
        _cachedOverlayExtentHeight = _swapChainExtent.height;
        _cachedMouseOverlayX = mouseX;
        _cachedMouseOverlayY = mouseY;
        _overlayDirty = false;
    }

    vkWaitForFences(_device, 1, &_inFlightFences[_currentFrame],
        VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        _device, _swapChain, UINT64_MAX,
        _imageAvailableSemaphores[_currentFrame],
        VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapChain(); return; }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR failed.");

    if (imageIndex >= _swapChainFramebuffers.size()) {
        return;
    }

    vkResetFences(_device, 1, &_inFlightFences[_currentFrame]);
    vkResetCommandBuffer(_commandBuffers[_currentFrame], 0);
    updateUniformBuffer(_currentFrame);
    recordCommandBuffer(_commandBuffers[_currentFrame], imageIndex);

    VkSemaphore waitSems[] = { _imageAvailableSemaphores[_currentFrame] };
    VkSemaphore signalSems[] = { _renderFinishedSemaphores[_currentFrame] };
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    };

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSems;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_commandBuffers[_currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSems;

    if (vkQueueSubmit(_graphicsQueue, 1, &submitInfo,
        _inFlightFences[_currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed.");
    _depthImageGpuReady = true;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSems;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapChain;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(_presentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR ||
        result == VK_SUBOPTIMAL_KHR || _framebufferResized) {
        _framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("vkQueuePresentKHR failed.");
    }

    _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    saveViewConfigIfChanged();
}

void VulkanWindow::enforceOrthoAspectFromWindow() {
    int w = 0;
    int h = 0;
    getQtFramebufferSize(w, h);
    if (w <= 1 || h <= 1) {
        return;
    }

    const float widthSpan = _ortho.y - _ortho.x;
    if (std::fabs(widthSpan) < 1e-6f) {
        return;
    }

    const float targetRatio = static_cast<float>(h) / static_cast<float>(w); // windowH / windowW
    const float targetHeightSpan = widthSpan * targetRatio;
    const float centerY = 0.5f * (_ortho.z + _ortho.w);
    _ortho.z = centerY - 0.5f * targetHeightSpan;
    _ortho.w = centerY + 0.5f * targetHeightSpan;
}

void VulkanWindow::setApplicationVertices(const std::vector<Vertex>& vertices) {
    _applicationVertices = vertices;
}

void VulkanWindow::setApplicationModelTransform(const glm::vec3& translation, const glm::vec3& scale) {
    _applicationModelTranslation = translation;
    _applicationModelScale = scale;
}

glm::ivec2 VulkanWindow::getWindowSize() const {
    if (_hostTopLevelWindow != nullptr) {
        const QSize sz = _hostTopLevelWindow->size();
        return glm::ivec2(std::max(1, sz.width()), std::max(1, sz.height()));
    }
    int w = 0;
    int h = 0;
    getQtWindowSize(w, h);
    if (w > 0 && h > 0) {
        return glm::ivec2(w, h);
    }
    return _viewport;
}

std::string VulkanWindow::getCursorReadoutText() {
    int ww = 0;
    int wh = 0;
    getQtWindowSize(ww, wh);
    const float safeW = (ww > 0) ? static_cast<float>(ww) : 1.0f;
    const float safeH = (wh > 0) ? static_cast<float>(wh) : 1.0f;
    const float clampedMouseX = static_cast<float>(std::clamp<double>(_lastX, 0.0, static_cast<double>(safeW)));
    const float clampedMouseY = static_cast<float>(std::clamp<double>(_lastY, 0.0, static_cast<double>(safeH)));
    const float mouseWorldX = _ortho.x + (clampedMouseX / safeW) * (_ortho.y - _ortho.x);
    const float mouseWorldY = _ortho.w - (clampedMouseY / safeH) * (_ortho.w - _ortho.z);

    char mouseBuf[256] = {};
    float depth01 = 1.0f;
    if (readDepthAtCursor(_lastX, _lastY, depth01)) {
        const glm::vec3 depthWorld = screenToWorldByDepth(_lastX, _lastY, depth01);
        if (!std::isfinite(depthWorld.x) || !std::isfinite(depthWorld.y) || !std::isfinite(depthWorld.z)) {
            std::snprintf(
                mouseBuf, sizeof(mouseBuf),
                "%.2f %.2f",
                static_cast<double>(mouseWorldX),
                static_cast<double>(mouseWorldY));
            return std::string(mouseBuf);
        }
        std::snprintf(
            mouseBuf, sizeof(mouseBuf),
            "%.2f %.2f %.2f, %.2f %.2f",
            static_cast<double>(depthWorld.x),
            static_cast<double>(depthWorld.y),
            static_cast<double>(depthWorld.z),
            static_cast<double>(mouseWorldX),
            static_cast<double>(mouseWorldY));
    }
    else {
        std::snprintf(
            mouseBuf, sizeof(mouseBuf),
            "%.2f %.2f",
            static_cast<double>(mouseWorldX),
            static_cast<double>(mouseWorldY));
    }
    return std::string(mouseBuf);
}

void VulkanWindow::setWindowSize(const glm::ivec2& size) {
    const int w = std::max(500, size.x);
    const int h = std::max(500, size.y);
    if (_hostTopLevelWindow != nullptr) {
        _hostTopLevelWindow->resize(w, h);
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }
    else if (_qtWindow != nullptr) {
        _qtWindow->resize(w, h);
    }
    syncViewportFromEmbeddedWindow();
}

void VulkanWindow::rebuildVertexBuffer() {
    if (_device == VK_NULL_HANDLE) {
        return;
    }
    vkDeviceWaitIdle(_device);
    if (_vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(_device, _vertexBuffer, nullptr);
        _vertexBuffer = VK_NULL_HANDLE;
    }
    if (_vertexBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(_device, _vertexBufferMemory, nullptr);
        _vertexBufferMemory = VK_NULL_HANDLE;
    }
    createVertexBuffer();
}
