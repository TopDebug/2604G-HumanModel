#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Eigen/Geometry>

#include <QEvent>
#include <QMouseEvent>
#include <QWidget>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>

static constexpr int MAX_FRAMES_IN_FLIGHT = 3;
static const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription d{};
        d.binding = 0;
        d.stride = sizeof(Vertex);
        d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> a{};
        a[0].binding = 0; a[0].location = 0;
        a[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        a[0].offset = offsetof(Vertex, pos);

        a[1].binding = 0; a[1].location = 1;
        a[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        a[1].offset = offsetof(Vertex, normal);

        a[2].binding = 0; a[2].location = 2;
        a[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        a[2].offset = offsetof(Vertex, color);
        return a;
    }
};

static const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
static constexpr bool ENABLE_VALIDATION = false;
#else
static constexpr bool ENABLE_VALIDATION = true;
#endif

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

class QWindow;
class QVulkanInstance;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* pData,
    void*)
{
    std::cerr << "[VK] " << pData->pMessage << '\n';
    return VK_FALSE;
}

/// QWidget hosting an embedded Vulkan QWindow plus swapchain / rendering logic.
class VulkanWindow : public QWidget {
public:
    explicit VulkanWindow(QWidget* parent = nullptr);
    ~VulkanWindow() override = default;

    void attachQtWindow(QWindow* window, QVulkanInstance* vulkanInstance);

    /// When set (e.g. top-level MainWindow), windowW/windowH resize this widget;
    /// the embedded Vulkan QWindow size is tracked separately in _viewport for input and overlays.
    void setHostTopLevelWindow(QWidget* host);

    void shutdownWindowSystem();

    bool handleQtEvent(QObject* watched, QEvent* event);

    void drawFrame();

protected:
    void setEmbeddedVulkanInstance(QVulkanInstance* inst);
    QWidget* _centralStack = nullptr;
    QWidget* _vkContainer = nullptr;
    QWindow* _vkWindow = nullptr;

    enum class MeshDisplayMode {
        Line,
        Face,
        LineAndFace
    };

    void rebuildVertexBuffer();
    void setApplicationVertices(const std::vector<Vertex>& vertices);
    void setApplicationModelTransform(const glm::vec3& translation, const glm::vec3& scale);
    glm::ivec2 getWindowSize() const;
    void setWindowSize(const glm::ivec2& size);

    void setRotation(const glm::vec2& rotationDeg) {
        auto wrapDegrees0To360 = [](float deg) {
            deg = std::fmod(deg, 360.0f);
            if (deg < 0.0f) {
                deg += 360.0f;
            }
            return deg;
        };
        _rotation = glm::vec2(wrapDegrees0To360(rotationDeg.x), wrapDegrees0To360(rotationDeg.y));
    }
    std::string getCursorReadoutText();
    void enforceOrthoAspectFromWindow();

    void cleanup();

    glm::vec3 _coordinate = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec2 _rotation = glm::vec2(0.0f, 0.0f);
    glm::vec4 _ortho = glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);
    glm::ivec2 _viewport = glm::ivec2(1200, 800);
    MeshDisplayMode _meshDisplayMode = MeshDisplayMode::LineAndFace;

    void markOverlayDirty();

    bool readDepthBufferFloat(std::vector<float>& outDepth, uint32_t& outW, uint32_t& outH);

private:
    QWidget* _hostTopLevelWindow = nullptr;
    QWindow* _qtWindow = nullptr;
    QVulkanInstance* _qVulkanInstance = nullptr;
    bool _ownsVkInstance = false;
    VkInstance               _instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             _surface = VK_NULL_HANDLE;
    VkPhysicalDevice         _physicalDevice = VK_NULL_HANDLE;
    VkDevice                 _device = VK_NULL_HANDLE;
    VkQueue                  _graphicsQueue = VK_NULL_HANDLE;
    VkQueue                  _presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR             _swapChain = VK_NULL_HANDLE;
    std::vector<VkImage>       _swapChainImages;
    VkFormat                   _swapChainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D                 _swapChainExtent{};
    std::vector<VkImageView>   _swapChainImageViews;
    std::vector<VkFramebuffer> _swapChainFramebuffers;

    VkImage        _depthImage = VK_NULL_HANDLE;
    VkDeviceMemory _depthImageMemory = VK_NULL_HANDLE;
    VkImageView    _depthImageView = VK_NULL_HANDLE;
    /// After at least one submitted frame, depth image is in ATTACHMENT_OPTIMAL; before that UNDEFINED.
    bool _depthImageGpuReady = false;
    VkDeviceSize   _copyRowPitchAlign = 1;
    VkDeviceSize   _copyOffsetAlign = 1;

    VkRenderPass          _renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      _pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            _trianglePipeline = VK_NULL_HANDLE;
    VkPipeline            _linePipeline = VK_NULL_HANDLE;
    VkPipeline            _rulerPipeline = VK_NULL_HANDLE;

    VkBuffer       _vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory _vertexBufferMemory = VK_NULL_HANDLE;
    uint32_t       _axisVertexCount = 0;
    uint32_t       _rulerVertexCount = 0;
    uint32_t       _applicationTriangleVertexCount = 0;
    uint32_t       _applicationLineVertexCount = 0;
    std::vector<Vertex> _applicationVertices;
    glm::vec3 _applicationModelTranslation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 _applicationModelScale = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec4 _cachedOrthoForOverlay = glm::vec4(0.0f);
    bool _hasCachedOrthoForOverlay = false;
    uint32_t _cachedOverlayExtentWidth = 0;
    uint32_t _cachedOverlayExtentHeight = 0;
    int _cachedMouseOverlayX = -1;
    int _cachedMouseOverlayY = -1;
    bool _overlayDirty = true;

    std::vector<VkBuffer>       _uniformBuffers;
    std::vector<VkDeviceMemory> _uniformBuffersMemory;
    std::vector<void*>          _uniformBuffersMapped;

    VkDescriptorPool             _descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> _descriptorSets;

    VkCommandPool                _commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> _commandBuffers;

    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
    std::vector<VkFence>     _inFlightFences;
    uint32_t _currentFrame = 0;
    bool     _framebufferResized = false;
    double _lastX = 0.0, _lastY = 0.0;
    bool   _leftMousePressed = false;
    bool _shiftPressedPrev = false;
    bool _ctrlPressedPrev = false;
    bool _altPressedPrev = false;
    std::int64_t _lastLeftClickTimestampMs = 0;
    glm::vec3 _lastSavedCoordinate = _coordinate;
    glm::vec2 _lastSavedRotation = _rotation;
    glm::vec4 _lastSavedOrtho = _ortho;
    glm::ivec2 _lastSavedWindowSize = _viewport;
    bool _hasSavedViewState = false;

    void getQtWindowSize(int& outW, int& outH) const;
    void getQtFramebufferSize(int& outW, int& outH) const;
    /// Swap chain pixel size used by vkCmdSetViewport / depth image; falls back to Qt framebuffer if unset.
    void getRenderTargetPixelSize(int& outW, int& outH) const;
    void syncViewportFromEmbeddedWindow();

    void cursorPosCallbackImpl(double xpos, double ypos);
    void mouseButtonCallbackImpl(Qt::MouseButton button, QEvent::Type type, Qt::KeyboardModifiers mods, double localX, double localY, long long timestampMs);
    void scrollCallbackImpl(float yoffset);

    glm::mat4 buildViewMatrix() const;
    glm::mat4 buildProjMatrix() const;
    glm::vec4 buildRulerOrtho() const;

    bool hasStencilComponent(VkFormat format) const;

    bool readDepthAtFramebufferPixel(uint32_t px, uint32_t py, float& outDepth01);

    bool readDepthAtCursor(double mouseX, double mouseY, float& outDepth01);

    glm::vec3 screenToWorldByDepth(double mouseX, double mouseY, float depth01) const;

    glm::vec3 worldToScreen(const glm::vec3& world) const;

    bool loadViewConfig();
    bool saveViewConfig() const;
    void saveViewConfigIfChanged();

    void initVulkan();

    void createInstance();

    bool checkValidationLayerSupport();

    std::vector<const char*> getRequiredExtensions();

    void fillDebugMessengerCI(VkDebugUtilsMessengerCreateInfoEXT& ci);

    void setupDebugMessenger();

    void createSurface();

    void pickPhysicalDevice();

    bool isDeviceSuitable(VkPhysicalDevice dev);

    bool checkDeviceExtensionSupport(VkPhysicalDevice dev);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev);

    void createLogicalDevice();

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& fmts);

    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes);

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps);

    void createSwapChain();

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

    void createImageViews();

    void createRenderPass();

    void createDescriptorSetLayout();

    void createGraphicsPipeline();
    void createRulerPipeline();

    void createCommandPool();

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

    VkFormat findDepthFormat();

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);

    void createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags props, VkImage& image, VkDeviceMemory& memory);

    void createDepthResources();

    void createFramebuffers();

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem);

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);

    void createVertexBuffer(bool probeDepthForRuler = true);

    void createUniformBuffers();

    void createDescriptorPool();

    void createDescriptorSets();

    void createCommandBuffers();

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

    void createSyncObjects();

    void updateUniformBuffer(uint32_t currentImage);

    void cleanupSwapChain();

    void recreateSwapChain();
};
