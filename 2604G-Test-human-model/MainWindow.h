#pragma once

#include "VulkanWindow.h"
#include "makehuman.h"

#include <QString>
#include <QVulkanInstance>
#include <Qt>

#include <cstdint>
#include <string>
#include <vector>

class QCloseEvent;
class QEvent;
class QResizeEvent;
class QShowEvent;

class QDoubleSpinBox;
class QLabel;
class QToolButton;
class QTimer;
class QAction;
class QMenuBar;
class QStackedWidget;

/// Application UI and mesh/view logic; inherits VulkanWindow (QWidget + Vulkan engine).
class MainWindow final : public VulkanWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    struct ViewState {
        glm::vec3 coordinate{};
        glm::vec2 rotation{};
        glm::vec4 ortho{};
        glm::ivec2 window{};
    };

    void shutdownBeforeQtTeardown();

    ViewState viewStateSnapshot() const;
    void applyViewState(const ViewState& s);

    void openObjFile(const std::string& path);
    void showCubeMesh();
    void setViewRotation(float degreesX, float degreesY);
    void runLscmAndShowMesh();

    void applyMakeHumanMesh(const makehuman::BodyParameters& params);

    bool captureDepthBufferFloat(std::vector<float>& outDepth, uint32_t& outW, uint32_t& outH);

    void updateReadoutLabel(const QString& text);
    void syncViewPanelFromApp();
    /** Each frame: refresh ortho / coordinate / rotation spins from engine unless that spin is being edited. */
    void syncOrthoSpinsFromAppUnlessFocused();

protected:
    void showEvent(QShowEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void attachAndInit();

    void buildMenuBar();
    void buildViewOverlayPanel();
    void applyViewPanelToApp();
    void showViewParametersPage();
    void activateCreateHumanPage();
    void syncHumanPanelFromParams();
    void applyHumanPanelToMesh();
    void updatePeekTabForSidePanel();

    void relayoutCentralOverlays();
    void logVulkanSurfaceSize(const char* reason) const;
    void scheduleViewDockAutoHide();
    void cancelViewDockAutoHide();
    void revealViewDockFromAutoHide();
    void applyAutoHideFromUi(bool on);

    void pumpFrame();

    std::vector<Vertex> buildCubeVertices() const;
    std::vector<Vertex> loadObjTriangles(const std::string& path) const;
    void showFlattenedMesh(const std::string& objPath);

    bool _showCube = false;

    QMenuBar* _menuBar = nullptr;

    QLabel* _panelStatusLabel = nullptr;

    QDoubleSpinBox* _coordX = nullptr;
    QDoubleSpinBox* _coordY = nullptr;
    QDoubleSpinBox* _coordZ = nullptr;
    QDoubleSpinBox* _rotX = nullptr;
    QDoubleSpinBox* _rotY = nullptr;
    QDoubleSpinBox* _orthoL = nullptr;
    QDoubleSpinBox* _orthoR = nullptr;
    QDoubleSpinBox* _orthoB = nullptr;
    QDoubleSpinBox* _orthoT = nullptr;
    QDoubleSpinBox* _windowW = nullptr;
    QDoubleSpinBox* _windowH = nullptr;

    QStackedWidget* _sidePanelStack = nullptr;
    QLabel* _overlayPanelTitleLabel = nullptr;
    QString _peekTabVerticalCaption = QStringLiteral("View");

    QDoubleSpinBox* _mhHeight = nullptr;
    QDoubleSpinBox* _mhGender = nullptr;
    QDoubleSpinBox* _mhChest = nullptr;
    QDoubleSpinBox* _mhWaist = nullptr;
    QDoubleSpinBox* _mhHips = nullptr;
    QDoubleSpinBox* _mhWeight = nullptr;
    QDoubleSpinBox* _mhArm = nullptr;
    QDoubleSpinBox* _mhLeg = nullptr;
    QDoubleSpinBox* _mhHead = nullptr;

    QWidget* _viewOverlayPanel = nullptr;
    QToolButton* _autoHidePeekTab = nullptr;
    QTimer* _viewAutoHideTimer = nullptr;
    bool _viewAutoHideEnabled = false;
    bool _panelOnLeft = true;
    int _viewPanelWidth = 220;
    QAction* _actAutoHidePanel = nullptr;
    QToolButton* _viewDockPinButton = nullptr;

    QTimer* _framePumpTimer = nullptr;
    bool _didAttachVulkan = false;

    QVulkanInstance _vulkanInstance;

    makehuman::BodyParameters _mhBodyParams{};
};
