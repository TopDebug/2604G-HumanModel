#include "MainWindow.h"
#include "Test.h"

#include <QApplication>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QEventLoop>
#include <QAction>
#include <QDialog>
#include <QAbstractSpinBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QMenu>
#include <QMenuBar>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPushButton>
#include <QPointer>
#include <QResizeEvent>
#include <QShowEvent>
#include <QScrollArea>
#include <QStackedWidget>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QTabWidget>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVersionNumber>
#include <QWidget>
#include <QIcon>
#include <QColor>
#include <QVulkanInstance>
#include <QWidget>
#include <QWindow>
#include <QFont>
#include <QFontMetrics>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

static double roundDepthWindow1(double v) {
    return std::round(v * 10.0) / 10.0;
}

static std::pair<double, double> computeDepthDataRange(const std::vector<float>& d) {
    if (d.empty()) {
        return { 0.0, 1.0 };
    }
    float mn = d[0];
    float mx = d[0];
    for (float v : d) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    if (!(static_cast<double>(mx) > static_cast<double>(mn) + 1e-20)) {
        mx = mn + 1e-3f;
    }
    return { static_cast<double>(mn), static_cast<double>(mx) };
}

/** Values below min map to black, above max to white; linear between. */
static void depthFloatToRgbaWindow(
    const std::vector<float>& depth,
    int iw,
    int ih,
    double wmin,
    double wmax,
    QImage& out) {
    double span = wmax - wmin;
    if (span <= 1e-20) {
        span = 1e-6;
    }
    out = QImage(iw, ih, QImage::Format_RGBA8888);
    for (int y = 0; y < ih; ++y) {
        uchar* row = out.scanLine(y);
        for (int x = 0; x < iw; ++x) {
            const float d = depth[static_cast<size_t>(y) * static_cast<size_t>(iw) + static_cast<size_t>(x)];
            double t = (static_cast<double>(d) - wmin) / span;
            t = std::clamp(t, 0.0, 1.0);
            const int g = static_cast<int>(std::lround(t * 255.0));
            const uchar u = static_cast<uchar>(std::clamp(g, 0, 255));
            row[x * 4 + 0] = u;
            row[x * 4 + 1] = u;
            row[x * 4 + 2] = u;
            row[x * 4 + 3] = 255;
        }
    }
}

/**
 * Depth contrast window on one slider:
 *   MIN — MAX   slide endpoints (full range, from data min/max ± pad)
 *   WL, WR      window positions on that slide (left/right contrast limits)
 * Slide spans data min/max ± pad, expanded to include WL (left) and WR (right) when outside that band.
 */
class DepthWindowStripWidget final : public QWidget {
public:
    /** Third arg false while dragging (fast); true on mouse release (apply full image). */
    using WindowCallback = std::function<void(double wmin, double wmax, bool committed)>;

    DepthWindowStripWidget(QWidget* parent, double dataMin, double dataMax, WindowCallback cb)
        : QWidget(parent)
        , _dataMin(dataMin)
        , _dataMax(dataMax)
        , _onWindowChanged(std::move(cb)) {
        setMinimumWidth(320);
        setFixedHeight(40);
        setMouseTracking(true);
        setFocusPolicy(Qt::StrongFocus);
        setAutoFillBackground(false);
    }

    void setWindowValues(double wmin, double wmax) {
        _WL = wmin;
        _WR = wmax;
        if (_WL > _WR) {
            std::swap(_WL, _WR);
        }
        update();
    }

    void setDataRange(double dmin, double dmax) {
        _dataMin = dmin;
        _dataMax = dmax;
        update();
    }

    bool isDragging() const {
        return _dragging;
    }

    /** Called at drag start so strip WL/WR match spin text (including uncommitted edits). */
    void setBeforeDragSync(std::function<void()> fn) {
        _beforeDragSync = std::move(fn);
    }

    QRect barRect() const {
        const int w = std::max(2, width() - 8);
        return QRect(4, 8, w, height() - 16);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);

        // Mid/dark gray slider (avoid near-white chrome).
        const QColor kFrame(0xA8, 0xA8, 0xA8);
        const QColor kTrack(0x8A, 0x8A, 0x8A);
        const QColor kTrackBorder(0x6A, 0x6A, 0x6A);
        const QColor kMark(0x2C, 0x2C, 0x2C);

        p.fillRect(rect(), kFrame);
        const QRect bar = barRect();
        p.setPen(QPen(kTrackBorder, 1));
        p.setBrush(kTrack);
        p.drawRect(bar);

        const double MINv = slideMin();
        const double MAXv = slideMax();
        if (!(MAXv > MINv)) {
            return;
        }

        auto xOf = [&](double v) -> int {
            double t = (v - MINv) / (MAXv - MINv);
            t = std::clamp(t, 0.0, 1.0);
            return bar.left() + static_cast<int>(std::lround(t * static_cast<double>(bar.width() - 1)));
        };

        double wl = _WL;
        double wr = _WR;
        if (wl > wr) {
            std::swap(wl, wr);
        }
        const int xWl = xOf(wl);
        const int xWr = xOf(wr);
        const int xl = std::min(xWl, xWr);
        const int xr = std::max(xWl, xWr);

        p.setPen(QPen(kMark, 2));
        p.drawLine(xl, bar.top(), xl, bar.bottom());
        p.drawLine(xr, bar.top(), xr, bar.bottom());
    }

    void mousePressEvent(QMouseEvent* e) override {
        if (e->button() != Qt::LeftButton) {
            return;
        }
        if (!barRect().contains(e->pos())) {
            return;
        }
        if (_beforeDragSync) {
            _beforeDragSync();
        }
        const QRect bar = barRect();
        const double MINv = slideMin();
        const double MAXv = slideMax();
        if (!(MAXv > MINv) || bar.width() < 2) {
            return;
        }
        auto xOf = [&](double v) -> int {
            double t = (v - MINv) / (MAXv - MINv);
            t = std::clamp(t, 0.0, 1.0);
            return bar.left() + static_cast<int>(std::lround(t * static_cast<double>(bar.width() - 1)));
        };
        double wl = _WL;
        double wr = _WR;
        if (wl > wr) {
            std::swap(wl, wr);
        }
        const int xWl = xOf(wl);
        const int xWr = xOf(wr);
        const int xl = std::min(xWl, xWr);
        const int xr = std::max(xWl, xWr);
        const int tol = 10;
        const int px = e->pos().x();
        if (px > xl + tol && px < xr - tol) {
            _dragMode = DragMode::Pan;
        }
        else {
            _dragMode = (std::abs(px - xWl) <= std::abs(px - xWr))
                ? DragMode::Left
                : DragMode::Right;
        }
        _dragging = true;
        _lastX = e->pos().x();
        grabMouse();
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        const QRect bar = barRect();
        const double MINv = slideMin();
        const double MAXv = slideMax();

        if (!_dragging) {
            if (!bar.contains(e->pos()) || !(MAXv > MINv) || bar.width() < 2) {
                setCursor(Qt::ArrowCursor);
                QWidget::mouseMoveEvent(e);
                return;
            }
            auto xOf = [&](double v) -> int {
                double t = (v - MINv) / (MAXv - MINv);
                t = std::clamp(t, 0.0, 1.0);
                return bar.left() + static_cast<int>(std::lround(t * static_cast<double>(bar.width() - 1)));
            };
            double wl = _WL;
            double wr = _WR;
            if (wl > wr) {
                std::swap(wl, wr);
            }
            const int xWl = xOf(wl);
            const int xWr = xOf(wr);
            const int xl = std::min(xWl, xWr);
            const int xr = std::max(xWl, xWr);
            const int tol = 10;
            const int px = e->pos().x();
            const bool nearHandle = (std::abs(px - xWl) <= tol) || (std::abs(px - xWr) <= tol);
            const bool inCenter = (px > xl + tol && px < xr - tol);
            setCursor((nearHandle || inCenter) ? Qt::SizeHorCursor : Qt::ArrowCursor);
            QWidget::mouseMoveEvent(e);
            return;
        }
        if (!(MAXv > MINv) || bar.width() < 2) {
            QWidget::mouseMoveEvent(e);
            return;
        }

        auto valAt = [&](int x) -> double {
            const int cx = std::clamp(x, bar.left(), bar.right());
            double t = static_cast<double>(cx - bar.left()) / static_cast<double>(std::max(1, bar.width() - 1));
            t = std::clamp(t, 0.0, 1.0);
            return MINv + t * (MAXv - MINv);
        };

        const double v = valAt(e->pos().x());
        const double v0 = valAt(_lastX);
        const double dv = v - v0;
        _lastX = e->pos().x();

        double wl = _WL;
        double wr = _WR;
        if (wl > wr) {
            std::swap(wl, wr);
        }
        if (_dragMode == DragMode::Pan) {
            const double span = MAXv - MINv;
            if (span <= 1e-18) {
                QWidget::mouseMoveEvent(e);
                return;
            }
            double w = std::max(wr - wl, 1e-9);
            if (w >= span - 1e-12) {
                wl = MINv;
                wr = MAXv;
            }
            else {
                w = std::min(w, span);
                wl = std::clamp(wl + dv, MINv, MAXv - w);
                wr = wl + w;
            }
        }
        else if (_dragMode == DragMode::Left) {
            wl = std::clamp(v, MINv, wr - 1e-9);
        }
        else {
            wr = std::clamp(v, wl + 1e-9, MAXv);
        }

        _WL = wl;
        _WR = wr;
        if (_onWindowChanged) {
            _onWindowChanged(wl, wr, false);
        }
        update();
        QWidget::mouseMoveEvent(e);
    }

    void mouseReleaseEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) {
            if (mouseGrabber() == this) {
                releaseMouse();
            }
            if (_dragging && _onWindowChanged) {
                double wl = _WL;
                double wr = _WR;
                if (wl > wr) {
                    std::swap(wl, wr);
                }
                _onWindowChanged(wl, wr, true);
            }
            _dragging = false;
            _dragMode = DragMode::None;
            setCursor(Qt::ArrowCursor);
        }
        QWidget::mouseReleaseEvent(e);
    }

private:
    enum class DragMode { None, Left, Right, Pan };

    static double axisPad(double dataMin, double dataMax) {
        return 0.05 * (dataMax - dataMin + 1e-9);
    }

    /**
     * Slide value range: default data min/max ± pad, expanded so WL is never left of the bar
     * and WR is never right of the bar (ordered WL ≤ WR).
     */
    double slideMin() const {
        const double pad = axisPad(_dataMin, _dataMax);
        const double baseMin = _dataMin - pad;
        double wl = _WL;
        double wr = _WR;
        if (wl > wr) {
            std::swap(wl, wr);
        }
        return std::min(baseMin, wl);
    }

    double slideMax() const {
        const double pad = axisPad(_dataMin, _dataMax);
        const double baseMax = _dataMax + pad;
        double wl = _WL;
        double wr = _WR;
        if (wl > wr) {
            std::swap(wl, wr);
        }
        return std::max(baseMax, wr);
    }

    double _dataMin = 0.0;
    double _dataMax = 1.0;
    double _WL = 0.0;
    double _WR = 1.0;
    bool _dragging = false;
    DragMode _dragMode = DragMode::None;
    int _lastX = 0;
    WindowCallback _onWindowChanged;
    std::function<void()> _beforeDragSync;
};

/** Depth preview: WL/WR spins, slider strip, hover readout (no caption), Save. */
class DepthBufferViewDialog final : public QDialog {
public:
    DepthBufferViewDialog(QWidget* parent, std::vector<float> depth, int iw, int ih, QString suggestedFileName)
        : QDialog(parent)
        , _depth(std::move(depth))
        , _iw(iw)
        , _ih(ih)
        , _suggestedFileName(std::move(suggestedFileName)) {
        setModal(false);
        setMinimumSize(400, 320);

        const auto rng = computeDepthDataRange(_depth);
        _dataMin = rng.first;
        _dataMax = rng.second;

        _windowLSpin = new QDoubleSpinBox(this);
        _windowRSpin = new QDoubleSpinBox(this);
        const QString spinGrayStyle = QStringLiteral(
            "QDoubleSpinBox { background-color: #E8E8E8; color: #1a1a1a; border: 1px solid #9E9E9E; "
            "border-radius: 2px; padding: 2px 6px; }");
        for (auto* s : { _windowLSpin, _windowRSpin }) {
            s->setRange(-1e9, 1e9);
            s->setDecimals(1);
            s->setSingleStep(0.1);
            s->setButtonSymbols(QAbstractSpinBox::NoButtons);
            s->setFixedWidth(100);
            s->setStyleSheet(spinGrayStyle);
        }
        _windowLSpin->setValue(roundDepthWindow1(static_cast<double>(rng.first)));
        _windowRSpin->setValue(roundDepthWindow1(static_cast<double>(rng.second)));

        auto* saveBtn = new QPushButton(QStringLiteral("Save…"), this);
        saveBtn->setAutoDefault(false);
        saveBtn->setDefault(false);
        saveBtn->setFocusPolicy(Qt::NoFocus);
        QObject::connect(saveBtn, &QPushButton::clicked, this, &DepthBufferViewDialog::savePng);

        _stripImageThrottleTimer = new QTimer(this);
        _stripImageThrottleTimer->setSingleShot(true);
        _stripImageThrottleTimer->setInterval(16);
        QObject::connect(_stripImageThrottleTimer, &QTimer::timeout, this, [this]() {
            if (!_stripImageDirtyFromDrag) {
                return;
            }
            _stripImageDirtyFromDrag = false;
            rebuildImage();
        });

        _strip = new DepthWindowStripWidget(this, _dataMin, _dataMax, [this](double wmin, double wmax, bool committed) {
            const QSignalBlocker b0(_windowLSpin);
            const QSignalBlocker b1(_windowRSpin);
            _windowLSpin->setValue(roundDepthWindow1(wmin));
            _windowRSpin->setValue(roundDepthWindow1(wmax));
            if (committed) {
                if (_stripImageThrottleTimer) {
                    _stripImageThrottleTimer->stop();
                }
                _stripImageDirtyFromDrag = false;
                rebuildImage();
                return;
            }
            _stripImageDirtyFromDrag = true;
            if (_stripImageThrottleTimer && !_stripImageThrottleTimer->isActive()) {
                _stripImageThrottleTimer->start();
            }
        });
        _strip->setBeforeDragSync([this]() {
            bool okL = true;
            bool okR = true;
            double wl = _windowLSpin->text().toDouble(&okL);
            double wr = _windowRSpin->text().toDouble(&okR);
            if (!okL) {
                wl = _windowLSpin->value();
            }
            if (!okR) {
                wr = _windowRSpin->value();
            }
            wl = roundDepthWindow1(wl);
            wr = roundDepthWindow1(wr);
            if (wl > wr) {
                std::swap(wl, wr);
            }
            {
                const QSignalBlocker b0(_windowLSpin);
                const QSignalBlocker b1(_windowRSpin);
                _windowLSpin->setValue(wl);
                _windowRSpin->setValue(wr);
            }
            _strip->setWindowValues(wl, wr);
        });
        _strip->setWindowValues(_windowLSpin->value(), _windowRSpin->value());

        _readoutLabel = new QLabel(QStringLiteral("—"), this);
        _readoutLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        _readoutLabel->setMinimumWidth(100);
        _readoutLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);

        _scroll = new QScrollArea(this);
        _scroll->setWidgetResizable(false);
        _scroll->setAlignment(Qt::AlignCenter);
        _scroll->setBackgroundRole(QPalette::Dark);
        _scroll->viewport()->setStyleSheet(QStringLiteral("background-color: #505050;"));

        _imageLabel = new QLabel;
        _imageLabel->setMouseTracking(true);
        _imageLabel->installEventFilter(this);
        _scroll->setWidget(_imageLabel);

        auto* controls = new QHBoxLayout;
        controls->setSpacing(8);
        controls->addWidget(new QLabel(QStringLiteral("WL"), this));
        controls->addWidget(_windowLSpin);
        controls->addSpacing(10);
        controls->addWidget(new QLabel(QStringLiteral("WR"), this));
        controls->addWidget(_windowRSpin);
        controls->addSpacing(14);
        controls->addWidget(_strip, 1);
        controls->addSpacing(16);
        controls->addWidget(_readoutLabel);
        controls->addWidget(saveBtn);

        auto* root = new QVBoxLayout(this);
        root->addWidget(_scroll, 1);
        root->addLayout(controls);

        QObject::connect(_windowLSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
            rebuildImage();
        });
        QObject::connect(_windowRSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
            rebuildImage();
        });
        for (auto* s : { _windowLSpin, _windowRSpin }) {
            if (QLineEdit* le = s->findChild<QLineEdit*>()) {
                QObject::connect(le, &QLineEdit::textChanged, this, [this]() {
                    syncSliderFromSpinText();
                });
            }
            QObject::connect(s, &QDoubleSpinBox::editingFinished, this, [this]() {
                rebuildImage();
            });
        }

        rebuildImage();

        const int guessW = std::clamp(_iw + 80, 480, 1000);
        const int guessH = std::clamp(_ih + 100, 360, 900);
        resize(guessW, guessH);
    }

    /** Replace captured depth (e.g. each time Export → Depth Buffer is chosen). */
    void reloadDepthData(std::vector<float>&& depth, int iw, int ih, QString suggestedFileName) {
        if (_stripImageThrottleTimer) {
            _stripImageThrottleTimer->stop();
        }
        _stripImageDirtyFromDrag = false;

        _depth = std::move(depth);
        _iw = iw;
        _ih = ih;
        _suggestedFileName = std::move(suggestedFileName);

        const auto rng = computeDepthDataRange(_depth);
        _dataMin = rng.first;
        _dataMax = rng.second;
        if (_strip) {
            _strip->setDataRange(_dataMin, _dataMax);
        }
        {
            const QSignalBlocker b0(_windowLSpin);
            const QSignalBlocker b1(_windowRSpin);
            _windowLSpin->setValue(roundDepthWindow1(static_cast<double>(rng.first)));
            _windowRSpin->setValue(roundDepthWindow1(static_cast<double>(rng.second)));
        }
        if (_strip) {
            _strip->setWindowValues(_windowLSpin->value(), _windowRSpin->value());
        }
        rebuildImage();
    }

protected:
    void closeEvent(QCloseEvent* e) override {
        hide();
        e->ignore();
    }

    bool eventFilter(QObject* watched, QEvent* event) override {
        if (watched != _imageLabel) {
            return QDialog::eventFilter(watched, event);
        }
        if (event->type() == QEvent::MouseMove) {
            const auto* me = static_cast<QMouseEvent*>(event);
            const int ix = me->pos().x();
            const int iy = me->pos().y();
            if (ix < 0 || iy < 0 || ix >= _iw || iy >= _ih) {
                _readoutLabel->setText(QStringLiteral("—"));
                return false;
            }
            const float pv = _depth[static_cast<size_t>(iy) * static_cast<size_t>(_iw) + static_cast<size_t>(ix)];
            _readoutLabel->setText(
                QStringLiteral("%1 %2 %3")
                    .arg(ix)
                    .arg(iy)
                    .arg(static_cast<double>(pv), 0, 'f', 1));
            return false;
        }
        if (event->type() == QEvent::Leave) {
            _readoutLabel->setText(QStringLiteral("—"));
            return false;
        }
        return QDialog::eventFilter(watched, event);
    }

private:
    /** Move strip handles while typing (before spin commits). One decimal, same as spins. */
    void syncSliderFromSpinText() {
        if (!_strip || _strip->isDragging()) {
            return;
        }
        bool okL = true;
        bool okR = true;
        double wl = _windowLSpin->text().toDouble(&okL);
        double wr = _windowRSpin->text().toDouble(&okR);
        if (!okL) {
            wl = _windowLSpin->value();
        }
        if (!okR) {
            wr = _windowRSpin->value();
        }
        wl = roundDepthWindow1(wl);
        wr = roundDepthWindow1(wr);
        if (wl > wr) {
            std::swap(wl, wr);
        }
        _strip->setWindowValues(wl, wr);
    }

    void rebuildImage() {
        double mn = roundDepthWindow1(_windowLSpin->value());
        double mx = roundDepthWindow1(_windowRSpin->value());
        if (!(mx > mn)) {
            mx = mn + 1e-9;
        }
        // Avoid feeding values back into the strip while it is actively dragging;
        // this removes visible flicker/jitter from redundant strip repaints.
        if (_strip && !_strip->isDragging()) {
            _strip->setWindowValues(mn, mx);
        }
        depthFloatToRgbaWindow(_depth, _iw, _ih, mn, mx, _image);
        _imageLabel->setPixmap(QPixmap::fromImage(_image));
        _imageLabel->resize(_iw, _ih);
    }

    void savePng() {
        auto restoreFocus = [this]() {
            QTimer::singleShot(0, this, [this]() {
                raise();
                activateWindow();
                if (_strip) {
                    _strip->setFocus(Qt::OtherFocusReason);
                }
            });
        };

        const QString path = QFileDialog::getSaveFileName(
            this,
            QStringLiteral("Save depth view"),
            _suggestedFileName,
            QStringLiteral("PNG (*.png);;All files (*)"));
        if (path.isEmpty()) {
            restoreFocus();
            return;
        }
        if (!_image.save(path, "PNG")) {
            QMessageBox::warning(this, QStringLiteral("Save"), QStringLiteral("Failed to save the image."));
        }
        restoreFocus();
    }

    std::vector<float> _depth;
    int _iw = 0;
    int _ih = 0;
    QString _suggestedFileName;
    QImage _image;
    QScrollArea* _scroll = nullptr;
    QLabel* _imageLabel = nullptr;
    QLabel* _readoutLabel = nullptr;
    QDoubleSpinBox* _windowLSpin = nullptr;
    QDoubleSpinBox* _windowRSpin = nullptr;
    DepthWindowStripWidget* _strip = nullptr;
    double _dataMin = 0.0;
    double _dataMax = 1.0;

    QTimer* _stripImageThrottleTimer = nullptr;
    bool _stripImageDirtyFromDrag = false;
};

void showDepthBufferViewer(QWidget* parent, MainWindow* win) {
    static QPointer<DepthBufferViewDialog> s_dialog;

    std::vector<float> depth;
    uint32_t w = 0;
    uint32_t h = 0;
    if (!win->captureDepthBufferFloat(depth, w, h) || w == 0 || h == 0) {
        QMessageBox::warning(
            parent,
            QStringLiteral("Depth Buffer"),
            QStringLiteral("Could not read the depth buffer (Vulkan may not be ready)."));
        return;
    }
    const QString suggested = QStringLiteral("DepthBuffer_%1_%2.png").arg(w).arg(h);
    const QString titleDetail = QStringLiteral("Depth Buffer — %1×%2").arg(w).arg(h);

    if (s_dialog) {
        s_dialog->reloadDepthData(std::move(depth), static_cast<int>(w), static_cast<int>(h), suggested);
        s_dialog->setWindowTitle(titleDetail);
        s_dialog->show();
        s_dialog->raise();
        s_dialog->activateWindow();
        return;
    }
    s_dialog = new DepthBufferViewDialog(parent, std::move(depth), static_cast<int>(w), static_cast<int>(h), suggested);
    s_dialog->setWindowTitle(titleDetail);
    s_dialog->show();
}

/** VS-style dock pin: vertical = pinned (stay open), horizontal = auto-hide. */
static QIcon makeDockPinIcon(bool autoHideOn) {
    constexpr QSize sz(16, 16);
    QPixmap pm(sz);
    pm.fill(Qt::transparent);
    QPainter painter(&pm);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor(255, 255, 255), 2.0));
    if (autoHideOn) {
        painter.drawLine(2, 8, 14, 8);
        painter.drawLine(2, 5, 2, 11);
    } else {
        painter.drawLine(8, 2, 8, 14);
        painter.drawLine(5, 4, 11, 4);
    }
    return QIcon(pm);
}

static void syncDockPinButtonState(QToolButton* pin, bool autoHideOn) {
    if (!pin) {
        return;
    }
    const QSignalBlocker b(pin);
    pin->setChecked(autoHideOn);
    pin->setIcon(makeDockPinIcon(autoHideOn));
    pin->setToolTip(autoHideOn ? QStringLiteral("Pin open (disable auto hide)")
                               : QStringLiteral("Auto Hide"));
}

/** Title strip for the side panel; pin toggles auto-hide. */
class ViewOverlayTitleBar final : public QWidget {
public:
    using AutoHideFn = std::function<void(bool)>;

    ViewOverlayTitleBar(const QString& titleText, AutoHideFn onAutoHide, QWidget* parent)
        : QWidget(parent)
        , _onAutoHide(std::move(onAutoHide)) {
        setObjectName(QStringLiteral("viewOverlayTitleBar"));
        setFixedHeight(28);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        setStyleSheet(QStringLiteral(
            "QWidget#viewOverlayTitleBar { background-color: #007ACC; border: none; }"
            "QWidget#viewOverlayTitleBar QToolButton { background: transparent; border: none; padding: 2px; }"
            "QWidget#viewOverlayTitleBar QToolButton:hover { background-color: rgba(255,255,255,0.14); "
            "border-radius: 2px; }"
            "QWidget#viewOverlayTitleBar QLabel { color: #FFFFFF; font-weight: 600; font-size: 13px; }"));

        auto* lay = new QHBoxLayout(this);
        lay->setContentsMargins(8, 0, 4, 0);
        lay->setSpacing(4);

        _title = new QLabel(titleText, this);
        _title->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        _title->setAttribute(Qt::WA_TransparentForMouseEvents, true);

        _pin = new QToolButton(this);
        _pin->setCheckable(true);
        _pin->setChecked(false);
        _pin->setAutoRaise(true);
        _pin->setFixedSize(26, 22);
        syncDockPinButtonState(_pin, false);
        QObject::connect(_pin, &QToolButton::toggled, this, [this](bool autoHideOn) {
            if (_onAutoHide) {
                _onAutoHide(autoHideOn);
            }
        });

        lay->addWidget(_title, 1);
        lay->addWidget(_pin, 0, Qt::AlignVCenter);
    }

    QLabel* titleLabel() const {
        return _title;
    }

    QToolButton* pinButton() const {
        return _pin;
    }

private:
    AutoHideFn _onAutoHide;
    QLabel* _title = nullptr;
    QToolButton* _pin = nullptr;
};

static bool pathLooksLikeObj(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".obj";
}

/** "View" drawn vertically for the narrow auto-hide peek strip (same orientation as before). */
static QPixmap makeVerticalCaptionPixmap(const QString& text, const QFont& font) {
    QFont f = font;
    const QFontMetrics fm(f);
    const int tw = fm.horizontalAdvance(text);
    const int th = fm.height();
    const int m = 6;
    QPixmap pm(th + 2 * m, tw + 2 * m);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::TextAntialiasing);
    p.setFont(f);
    p.setPen(QColor(255, 255, 255));
    p.translate(pm.width() / 2.0, pm.height() / 2.0);
    p.rotate(-90);
    p.drawText(QRect(-tw / 2, -th / 2, tw, th), Qt::AlignCenter, text);
    return pm;
}

} // namespace

MainWindow::~MainWindow() {
    if (_framePumpTimer) {
        _framePumpTimer->stop();
    }
    shutdownBeforeQtTeardown();
}

MainWindow::MainWindow(QWidget* parent)
    : VulkanWindow(parent) {
    _vulkanInstance.setApiVersion(QVersionNumber(1, 2, 0));
    if (!_vulkanInstance.create()) {
        throw std::runtime_error("QVulkanInstance::create() failed (install a Vulkan driver and dev packages).");
    }
    setEmbeddedVulkanInstance(&_vulkanInstance);

    setWindowTitle(QStringLiteral("MINGJIE2026@GMAIL.COM"));
    // QWidget (unlike QMainWindow) does not quit the application when closed unless this is set.
    setAttribute(Qt::WA_QuitOnClose, true);

    auto* rootLay = new QVBoxLayout(this);
    rootLay->setContentsMargins(0, 0, 0, 0);
    rootLay->setSpacing(0);
    _menuBar = new QMenuBar(this);
    rootLay->addWidget(_menuBar);
    rootLay->addWidget(_centralStack, 1);

    // createWindowContainer embeds a native Vulkan window; on X11/Wayland it usually paints above
    // overlapping Qt siblings, so an overlaid panel is invisible. We lay out the panel / peek tab in a
    // non-overlapping strip and shrink the VK container to the remaining rect (see relayoutCentralOverlays).

    _autoHidePeekTab = new QToolButton(_centralStack);
    _autoHidePeekTab->setObjectName(QStringLiteral("viewAutoHidePeekTab"));
    _autoHidePeekTab->setText(QString());
    _autoHidePeekTab->setToolTip(QStringLiteral("Show side panel (auto-hide)"));
    _autoHidePeekTab->setAutoRaise(true);
    _autoHidePeekTab->setCheckable(false);
    _autoHidePeekTab->setToolButtonStyle(Qt::ToolButtonIconOnly);
    _autoHidePeekTab->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    _autoHidePeekTab->hide();
    _autoHidePeekTab->setStyleSheet(QStringLiteral(
        "QToolButton#viewAutoHidePeekTab {"
        "  background-color: #3C3C3C;"
        "  border: 1px solid #555;"
        "  border-radius: 0px;"
        "  padding: 2px 1px;"
        "}"
        "QToolButton#viewAutoHidePeekTab:hover { background-color: #505050; }"));
    QObject::connect(_autoHidePeekTab, &QToolButton::clicked, this, [this]() {
        revealViewDockFromAutoHide();
    });
    _autoHidePeekTab->installEventFilter(this);

    _viewAutoHideTimer = new QTimer(this);
    _viewAutoHideTimer->setSingleShot(true);
    _viewAutoHideTimer->setInterval(350);
    QObject::connect(_viewAutoHideTimer, &QTimer::timeout, this, [this]() {
        if (!_viewAutoHideEnabled || !_viewOverlayPanel) {
            return;
        }
        if (!_viewOverlayPanel->isVisible()) {
            return;
        }
        _viewOverlayPanel->hide();
        _autoHidePeekTab->show();
        relayoutCentralOverlays();
    });

    buildViewOverlayPanel();
    buildMenuBar();
    updatePeekTabForSidePanel();

    resize(1280, 840);
    relayoutCentralOverlays();

    setHostTopLevelWindow(this);
    // Mouse/wheel hit the QWidget container; keys/resize may still reach the QWindow.
    _vkContainer->installEventFilter(this);
    _vkWindow->installEventFilter(this);
}

void MainWindow::buildMenuBar() {
    if (!_menuBar) {
        return;
    }
    // VS Code–like: 1px hairline, subtle gray, full width; vertical gap is margin, not thick bar.
    _menuBar->setStyleSheet(QStringLiteral(
        "QMenu::separator {"
        "  height: 1px;"
        "  margin: 7px 0px;"
        "  padding: 0px;"
        "  background: #3E3E42;"
        "  border: none;"
        "}"));

    QMenu* fileMenu = _menuBar->addMenu(QStringLiteral("File"));
    QObject::connect(fileMenu->addAction(QStringLiteral("Open…")), &QAction::triggered, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("Open OBJ"),
            QString(),
            QStringLiteral("Wavefront OBJ (*.obj);;All files (*)"));
        if (path.isEmpty()) {
            return;
        }
        if (!path.endsWith(QStringLiteral(".obj"), Qt::CaseInsensitive)) {
            QMessageBox::warning(this, QStringLiteral("Open"), QStringLiteral("Please select an .obj file."));
            return;
        }
        openObjFile(path.toStdString());
    });

    QMenu* viewMenu = _menuBar->addMenu(QStringLiteral("View"));
    QObject::connect(viewMenu->addAction(QStringLiteral("Front View")), &QAction::triggered, this, [this]() {
        setViewRotation(0.0f, 0.0f);
    });
    QObject::connect(viewMenu->addAction(QStringLiteral("Right View")), &QAction::triggered, this, [this]() {
        setViewRotation(0.0f, -90.0f);
    });
    QObject::connect(viewMenu->addAction(QStringLiteral("Top View")), &QAction::triggered, this, [this]() {
        setViewRotation(90.0f, 0.0f);
    });

    viewMenu->addSeparator();
    QObject::connect(viewMenu->addAction(QStringLiteral("View parameters")), &QAction::triggered, this, [this]() {
        showViewParametersPage();
    });

    viewMenu->addSeparator();
    QObject::connect(viewMenu->addAction(QStringLiteral("Panel Left")), &QAction::triggered, this, [this]() {
        cancelViewDockAutoHide();
        _panelOnLeft = true;
        relayoutCentralOverlays();
    });
    QObject::connect(viewMenu->addAction(QStringLiteral("Panel Right")), &QAction::triggered, this, [this]() {
        cancelViewDockAutoHide();
        _panelOnLeft = false;
        relayoutCentralOverlays();
    });

    QMenu* shapeMenu = _menuBar->addMenu(QStringLiteral("Shape"));
    QObject::connect(shapeMenu->addAction(QStringLiteral("Cube")), &QAction::triggered, this, [this]() {
        showCubeMesh();
    });

    QMenu* humanMenu = _menuBar->addMenu(QStringLiteral("Human"));
    QObject::connect(humanMenu->addAction(QStringLiteral("Make Human")), &QAction::triggered, this, [this]() {
        activateCreateHumanPage();
    });

    QMenu* exportMenu = _menuBar->addMenu(QStringLiteral("Tools"));
    QObject::connect(exportMenu->addAction(QStringLiteral("Depth Buffer")), &QAction::triggered, this, [this]() {
        showDepthBufferViewer(this, this);
    });

    QMenu* testMenu = _menuBar->addMenu(QStringLiteral("Test"));
    QObject::connect(testMenu->addAction(QStringLiteral("lscm")), &QAction::triggered, this, [this]() {
        runLscmAndShowMesh();
    });
}

void MainWindow::buildViewOverlayPanel() {
    _viewOverlayPanel = new QWidget(_centralStack);
    _viewOverlayPanel->setObjectName(QStringLiteral("viewParametersOverlay"));
    _viewOverlayPanel->setMouseTracking(true);
    _viewOverlayPanel->setAttribute(Qt::WA_Hover, true);
    _viewOverlayPanel->setStyleSheet(QStringLiteral(
        "QWidget#viewParametersOverlay {"
        "  background-color: #252526;"
        "  border: 1px solid #3F3F46;"
        "}"));

    auto* rootLay = new QVBoxLayout(_viewOverlayPanel);
    rootLay->setContentsMargins(0, 0, 0, 0);
    rootLay->setSpacing(0);

    auto* titleBar = new ViewOverlayTitleBar(
        QStringLiteral("View"),
        [this](bool on) { applyAutoHideFromUi(on); },
        _viewOverlayPanel);
    _viewDockPinButton = titleBar->pinButton();
    _overlayPanelTitleLabel = titleBar->titleLabel();
    rootLay->addWidget(titleBar);

    _sidePanelStack = new QStackedWidget(_viewOverlayPanel);
    _sidePanelStack->setMouseTracking(true);

    auto* viewPage = new QWidget(_sidePanelStack);
    auto* viewPageLay = new QVBoxLayout(viewPage);
    viewPageLay->setContentsMargins(0, 0, 0, 0);
    viewPageLay->setSpacing(0);

    auto* panel = new QWidget(viewPage);
    panel->setObjectName(QStringLiteral("viewParamsContent"));
    panel->setMouseTracking(true);
    panel->setAttribute(Qt::WA_Hover, true);
    panel->setAttribute(Qt::WA_TransparentForMouseEvents, false);
    panel->setAttribute(Qt::WA_StyledBackground, true);
    panel->setStyleSheet(QStringLiteral(
        "QWidget#viewParamsContent { background-color: #252526; }"
        "QWidget#viewParamsContent QDoubleSpinBox { background-color: #3C3C3C; color: #E0E0E0; border: 1px solid #555; }"
        "QWidget#viewParamsContent QLabel { background: transparent; color: #E0E0E0; }"));
    auto* form = new QFormLayout(panel);
    form->setContentsMargins(8, 8, 8, 8);

    auto makeSpin = [](double lo, double hi, int decimals) {
        auto* s = new QDoubleSpinBox();
        s->setRange(lo, hi);
        s->setDecimals(decimals);
        s->setSingleStep(0.1);
        s->setButtonSymbols(QAbstractSpinBox::NoButtons);
        s->setFixedWidth(100);
        // No valueChanged (→ apply) while typing; commit on Enter or focus out only.
        s->setKeyboardTracking(false);
        return s;
    };

    _coordX = makeSpin(-1e6, 1e6, 2);
    _coordY = makeSpin(-1e6, 1e6, 2);
    _coordZ = makeSpin(-1e6, 1e6, 2);
    _rotX = makeSpin(0.0, 360.0, 2);
    _rotY = makeSpin(0.0, 360.0, 2);
    _orthoL = makeSpin(-1e6, 1e6, 2);
    _orthoR = makeSpin(-1e6, 1e6, 2);
    _orthoB = makeSpin(-1e6, 1e6, 2);
    _orthoT = makeSpin(-1e6, 1e6, 2);
    _windowW = makeSpin(500, 16000, 0);
    _windowH = makeSpin(500, 16000, 0);

    form->addRow(QStringLiteral("coordinateX"), _coordX);
    form->addRow(QStringLiteral("coordinateY"), _coordY);
    form->addRow(QStringLiteral("coordinateZ"), _coordZ);
    form->addRow(QStringLiteral("rotateX"), _rotX);
    form->addRow(QStringLiteral("rotateY"), _rotY);
    form->addRow(QStringLiteral("orthoL"), _orthoL);
    form->addRow(QStringLiteral("orthoR"), _orthoR);
    form->addRow(QStringLiteral("orthoB"), _orthoB);
    form->addRow(QStringLiteral("orthoT"), _orthoT);
    form->addRow(QStringLiteral("windowW"), _windowW);
    form->addRow(QStringLiteral("windowH"), _windowH);

    auto apply = [this]() { applyViewPanelToApp(); };
    for (auto* s : { _coordX, _coordY, _coordZ, _rotX, _rotY, _orthoL, _orthoR, _orthoB, _orthoT, _windowW, _windowH }) {
        s->setMouseTracking(true);
        s->setAttribute(Qt::WA_Hover, true);
        s->setAttribute(Qt::WA_TransparentForMouseEvents, false);
        QObject::connect(s, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [apply](double) { apply(); });
    }

    viewPageLay->addWidget(panel, 1);
    _sidePanelStack->addWidget(viewPage);

    auto* humanPage = new QWidget(_sidePanelStack);
    humanPage->setObjectName(QStringLiteral("humanParamsContent"));
    humanPage->setMouseTracking(true);
    humanPage->setAttribute(Qt::WA_Hover, true);
    humanPage->setAttribute(Qt::WA_TransparentForMouseEvents, false);
    humanPage->setAttribute(Qt::WA_StyledBackground, true);
    humanPage->setStyleSheet(QStringLiteral(
        "QWidget#humanParamsContent { background-color: #252526; }"
        "QWidget#humanParamsContent QDoubleSpinBox { background-color: #3C3C3C; color: #E0E0E0; border: 1px solid #555; }"
        "QWidget#humanParamsContent QLabel { background: transparent; color: #E0E0E0; }"));
    auto* humanRootLay = new QVBoxLayout(humanPage);
    humanRootLay->setContentsMargins(0, 0, 0, 0);
    humanRootLay->setSpacing(6);
    auto* humanForm = new QFormLayout();
    humanForm->setContentsMargins(8, 8, 8, 4);

    auto mkHumanSpin = [](double lo, double hi, int dec) {
        auto* s = new QDoubleSpinBox();
        s->setRange(lo, hi);
        s->setDecimals(dec);
        s->setSingleStep(dec >= 2 ? 0.05 : 0.5);
        s->setButtonSymbols(QAbstractSpinBox::NoButtons);
        s->setFixedWidth(100);
        s->setKeyboardTracking(false);
        return s;
    };

    _mhHeight = mkHumanSpin(1.2, 2.1, 2);
    _mhGender = mkHumanSpin(0.0, 1.0, 2);
    _mhChest = mkHumanSpin(0.5, 1.5, 2);
    _mhWaist = mkHumanSpin(0.5, 1.5, 2);
    _mhHips = mkHumanSpin(0.5, 1.5, 2);
    _mhWeight = mkHumanSpin(0.5, 1.5, 2);
    _mhArm = mkHumanSpin(0.75, 1.25, 2);
    _mhLeg = mkHumanSpin(0.75, 1.25, 2);
    _mhHead = mkHumanSpin(0.75, 1.25, 2);

    humanForm->addRow(QStringLiteral("Height (m)"), _mhHeight);
    humanForm->addRow(QStringLiteral("Gender (0 female, 1 male)"), _mhGender);
    humanForm->addRow(QStringLiteral("Chest"), _mhChest);
    humanForm->addRow(QStringLiteral("Waist"), _mhWaist);
    humanForm->addRow(QStringLiteral("Hips"), _mhHips);
    humanForm->addRow(QStringLiteral("Weight / bulk"), _mhWeight);
    humanForm->addRow(QStringLiteral("Arm length"), _mhArm);
    humanForm->addRow(QStringLiteral("Leg length"), _mhLeg);
    humanForm->addRow(QStringLiteral("Head scale"), _mhHead);
    humanRootLay->addLayout(humanForm);

    auto* restoreDefaults = new QPushButton(QStringLiteral("Restore defaults"), humanPage);
    restoreDefaults->setAutoDefault(false);
    restoreDefaults->setDefault(false);
    QObject::connect(restoreDefaults, &QPushButton::clicked, this, [this]() {
        _mhBodyParams = makehuman::BodyParameters{};
        syncHumanPanelFromParams();
        applyMakeHumanMesh(_mhBodyParams);
    });
    humanRootLay->addWidget(restoreDefaults);
    humanRootLay->addStretch(1);

    for (auto* s : { _mhHeight, _mhGender, _mhChest, _mhWaist, _mhHips, _mhWeight, _mhArm, _mhLeg, _mhHead }) {
        s->setMouseTracking(true);
        s->setAttribute(Qt::WA_Hover, true);
        s->setAttribute(Qt::WA_TransparentForMouseEvents, false);
        QObject::connect(s, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) { applyHumanPanelToMesh(); });
    }

    _sidePanelStack->addWidget(humanPage);
    _sidePanelStack->setCurrentIndex(0);

    rootLay->addWidget(_sidePanelStack, 1);

    auto* statusStrip = new QWidget(_viewOverlayPanel);
    statusStrip->setObjectName(QStringLiteral("viewPanelStatusBar"));
    statusStrip->setFixedHeight(24);
    statusStrip->setStyleSheet(QStringLiteral(
        "QWidget#viewPanelStatusBar { background-color: #1E1E1E; border-top: 1px solid #3F3F46; }"
        "QWidget#viewPanelStatusBar QLabel { color: #CCCCCC; font-size: 11px; padding: 2px 6px; background: transparent; }"));
    auto* statusLay = new QHBoxLayout(statusStrip);
    statusLay->setContentsMargins(0, 0, 0, 0);
    statusLay->setSpacing(0);
    _panelStatusLabel = new QLabel(QStringLiteral(" "), statusStrip);
    _panelStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    _panelStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    _panelStatusLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    statusLay->addWidget(_panelStatusLabel, 1);
    rootLay->addWidget(statusStrip, 0);

    _viewOverlayPanel->installEventFilter(this);
    _viewOverlayPanel->show();
    _viewOverlayPanel->raise();

    syncHumanPanelFromParams();
}

void MainWindow::syncViewPanelFromApp() {
    const auto s = viewStateSnapshot();
    const QSignalBlocker bx(_coordX);
    const QSignalBlocker by(_coordY);
    const QSignalBlocker bz(_coordZ);
    const QSignalBlocker brx(_rotX);
    const QSignalBlocker bry(_rotY);
    const QSignalBlocker bol(_orthoL);
    const QSignalBlocker bor(_orthoR);
    const QSignalBlocker bob(_orthoB);
    const QSignalBlocker bot(_orthoT);
    const QSignalBlocker bvw(_windowW);
    const QSignalBlocker bvh(_windowH);
    _coordX->setValue(s.coordinate.x);
    _coordY->setValue(s.coordinate.y);
    _coordZ->setValue(s.coordinate.z);
    _rotX->setValue(s.rotation.x);
    _rotY->setValue(s.rotation.y);
    _orthoL->setValue(s.ortho.x);
    _orthoR->setValue(s.ortho.y);
    _orthoB->setValue(s.ortho.z);
    _orthoT->setValue(s.ortho.w);
    _windowW->setValue(s.window.x);
    _windowH->setValue(s.window.y);
}

static bool doubleSpinOrChildHasFocus(const QDoubleSpinBox* s) {
    if (!s) {
        return false;
    }
    QWidget* fw = QApplication::focusWidget();
    return fw != nullptr && (fw == s || s->isAncestorOf(fw));
}

void MainWindow::syncOrthoSpinsFromAppUnlessFocused() {
    if (!_coordX || !_coordY || !_coordZ || !_rotX || !_rotY
        || !_orthoL || !_orthoR || !_orthoB || !_orthoT) {
        return;
    }
    const auto s = viewStateSnapshot();
    auto sync = [](QDoubleSpinBox* box, double v) {
        if (!box || doubleSpinOrChildHasFocus(box)) {
            return;
        }
        const QSignalBlocker b(box);
        box->setValue(v);
    };
    sync(_coordX, static_cast<double>(s.coordinate.x));
    sync(_coordY, static_cast<double>(s.coordinate.y));
    sync(_coordZ, static_cast<double>(s.coordinate.z));
    sync(_rotX, static_cast<double>(s.rotation.x));
    sync(_rotY, static_cast<double>(s.rotation.y));
    sync(_orthoL, static_cast<double>(s.ortho.x));
    sync(_orthoR, static_cast<double>(s.ortho.y));
    sync(_orthoB, static_cast<double>(s.ortho.z));
    sync(_orthoT, static_cast<double>(s.ortho.w));
}

void MainWindow::applyViewPanelToApp() {
    ViewState s;
    s.coordinate = { static_cast<float>(_coordX->value()), static_cast<float>(_coordY->value()), static_cast<float>(_coordZ->value()) };
    s.rotation = { static_cast<float>(_rotX->value()), static_cast<float>(_rotY->value()) };
    s.ortho = { static_cast<float>(_orthoL->value()), static_cast<float>(_orthoR->value()),
        static_cast<float>(_orthoB->value()), static_cast<float>(_orthoT->value()) };
    s.window = { static_cast<int>(_windowW->value()), static_cast<int>(_windowH->value()) };
    applyViewState(s);
}

void MainWindow::updatePeekTabForSidePanel() {
    if (!_autoHidePeekTab || !_menuBar) {
        return;
    }
    const QPixmap cap = makeVerticalCaptionPixmap(_peekTabVerticalCaption, _menuBar->font());
    _autoHidePeekTab->setIcon(QIcon(cap));
    _autoHidePeekTab->setIconSize(cap.size());
    _autoHidePeekTab->setFixedWidth(std::max(22, cap.width() + 4));
}

void MainWindow::showViewParametersPage() {
    if (!_sidePanelStack || !_overlayPanelTitleLabel) {
        return;
    }
    _peekTabVerticalCaption = QStringLiteral("View");
    updatePeekTabForSidePanel();
    _overlayPanelTitleLabel->setText(QStringLiteral("View"));
    _sidePanelStack->setCurrentIndex(0);
    _viewPanelWidth = 220;
    syncViewPanelFromApp();
    cancelViewDockAutoHide();
    if (_viewOverlayPanel && !_viewOverlayPanel->isVisible()) {
        _viewOverlayPanel->show();
    }
    relayoutCentralOverlays();
}

void MainWindow::activateCreateHumanPage() {
    if (!_sidePanelStack || !_overlayPanelTitleLabel) {
        return;
    }
    _peekTabVerticalCaption = QStringLiteral("Human");
    updatePeekTabForSidePanel();
    _overlayPanelTitleLabel->setText(QStringLiteral("Human"));
    _sidePanelStack->setCurrentIndex(1);
    _viewPanelWidth = 248;
    syncHumanPanelFromParams();
    applyMakeHumanMesh(_mhBodyParams);
    cancelViewDockAutoHide();
    if (_viewOverlayPanel && !_viewOverlayPanel->isVisible()) {
        _viewOverlayPanel->show();
    }
    relayoutCentralOverlays();
}

void MainWindow::syncHumanPanelFromParams() {
    if (!_mhHeight || !_mhGender || !_mhChest || !_mhWaist || !_mhHips || !_mhWeight || !_mhArm || !_mhLeg || !_mhHead) {
        return;
    }
    const makehuman::BodyParameters& p = _mhBodyParams;
    const QSignalBlocker bh(_mhHeight);
    const QSignalBlocker bg(_mhGender);
    const QSignalBlocker bc(_mhChest);
    const QSignalBlocker bw(_mhWaist);
    const QSignalBlocker bhp(_mhHips);
    const QSignalBlocker bwt(_mhWeight);
    const QSignalBlocker ba(_mhArm);
    const QSignalBlocker bl(_mhLeg);
    const QSignalBlocker bhd(_mhHead);
    _mhHeight->setValue(p.heightMeters);
    _mhGender->setValue(p.genderBlend);
    _mhChest->setValue(p.chest);
    _mhWaist->setValue(p.waist);
    _mhHips->setValue(p.hips);
    _mhWeight->setValue(p.weight);
    _mhArm->setValue(p.armLength);
    _mhLeg->setValue(p.legLength);
    _mhHead->setValue(p.headScale);
}

void MainWindow::applyHumanPanelToMesh() {
    if (!_mhHeight || !_mhGender || !_mhChest || !_mhWaist || !_mhHips || !_mhWeight || !_mhArm || !_mhLeg || !_mhHead) {
        return;
    }
    makehuman::BodyParameters p;
    p.heightMeters = static_cast<float>(_mhHeight->value());
    p.genderBlend = static_cast<float>(_mhGender->value());
    p.chest = static_cast<float>(_mhChest->value());
    p.waist = static_cast<float>(_mhWaist->value());
    p.hips = static_cast<float>(_mhHips->value());
    p.weight = static_cast<float>(_mhWeight->value());
    p.armLength = static_cast<float>(_mhArm->value());
    p.legLength = static_cast<float>(_mhLeg->value());
    p.headScale = static_cast<float>(_mhHead->value());
    _mhBodyParams = p;
    applyMakeHumanMesh(p);
}

void MainWindow::updateReadoutLabel(const QString& text) {
    if (_panelStatusLabel) {
        _panelStatusLabel->setText(text);
    }
}

void MainWindow::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    if (_didAttachVulkan) {
        return;
    }
    QCoreApplication::processEvents(QEventLoop::AllEvents);
    attachAndInit();
    syncViewPanelFromApp();
    _didAttachVulkan = true;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (_framePumpTimer) {
        _framePumpTimer->stop();
    }
    shutdownBeforeQtTeardown();
    QWidget::closeEvent(event);
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    relayoutCentralOverlays();
}

void MainWindow::logVulkanSurfaceSize(const char* reason) const {
    QWindow* vk = _vkWindow;
    if (!vk) {
        std::printf("[Vulkan window] %s: (null QWindow)\n", reason);
        std::fflush(stdout);
        return;
    }
    const int lw = vk->width();
    const int lh = vk->height();
    const qreal dpr = vk->devicePixelRatio();
    const int fw = static_cast<int>(std::lround(static_cast<qreal>(lw) * dpr));
    const int fh = static_cast<int>(std::lround(static_cast<qreal>(lh) * dpr));
    std::printf(
        "[Vulkan window] %s: logical %dx%d, dpr %.3f, framebuffer ~%dx%d\n",
        reason,
        lw,
        lh,
        dpr,
        fw,
        fh);
    std::fflush(stdout);
}

void MainWindow::relayoutCentralOverlays() {
    if (!_centralStack || !_vkContainer) {
        return;
    }
    const int W = _centralStack->width();
    const int H = _centralStack->height();
    if (W <= 0 || H <= 0) {
        return;
    }

    const int pw = _viewPanelWidth;
    const int edge = (_autoHidePeekTab && _autoHidePeekTab->isVisible())
        ? std::max(22, _autoHidePeekTab->width())
        : 22;
    int vkX = 0;
    int vkY = 0;
    int vkW = W;
    int vkH = H;

    if (_viewOverlayPanel && _viewOverlayPanel->isVisible()) {
        const int stripW = std::min(pw, std::max(1, W - 1));
        if (_panelOnLeft) {
            _viewOverlayPanel->setGeometry(0, 0, stripW, H);
            vkX = stripW;
            vkW = W - stripW;
        } else {
            _viewOverlayPanel->setGeometry(W - stripW, 0, stripW, H);
            vkX = 0;
            vkW = W - stripW;
        }
        _viewOverlayPanel->raise();
    } else if (_autoHidePeekTab && _autoHidePeekTab->isVisible()) {
        if (_panelOnLeft) {
            _autoHidePeekTab->setGeometry(0, 0, edge, H);
            vkX = edge;
            vkW = W - edge;
        } else {
            _autoHidePeekTab->setGeometry(W - edge, 0, edge, H);
            vkX = 0;
            vkW = W - edge;
        }
        _autoHidePeekTab->raise();
    }

    vkW = std::max(1, vkW);
    vkH = std::max(1, vkH);
    _vkContainer->setGeometry(vkX, vkY, vkW, vkH);
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    const QEvent::Type t = event->type();
    const bool mouseOrWheel = (t == QEvent::MouseMove || t == QEvent::MouseButtonPress
        || t == QEvent::MouseButtonRelease || t == QEvent::Wheel);

    if (watched == _vkContainer) {
        if (mouseOrWheel || t == QEvent::Resize) {
            handleQtEvent(watched, event);
        }
    }
    else if (watched == _vkWindow) {
        // Some platforms deliver embedded-Vulkan input to the QWindow, not the container.
        if (mouseOrWheel || t == QEvent::KeyPress || t == QEvent::Resize) {
            handleQtEvent(watched, event);
        }
    }
    if (_viewOverlayPanel && watched == _viewOverlayPanel) {
        if (event->type() == QEvent::Show || event->type() == QEvent::Hide) {
            cancelViewDockAutoHide();
            const bool visible = (event->type() == QEvent::Show);
            if (visible) {
                _autoHidePeekTab->hide();
            } else if (_viewAutoHideEnabled) {
                _autoHidePeekTab->show();
            } else {
                _autoHidePeekTab->hide();
            }
            relayoutCentralOverlays();
            logVulkanSurfaceSize(visible ? "view panel shown" : "view panel hidden");
        }
        if (_viewAutoHideEnabled) {
            if (event->type() == QEvent::Leave) {
                scheduleViewDockAutoHide();
            } else if (event->type() == QEvent::Enter) {
                cancelViewDockAutoHide();
            }
        }
    }
    if (_autoHidePeekTab && _viewAutoHideEnabled && watched == _autoHidePeekTab) {
        if (event->type() == QEvent::Enter) {
            revealViewDockFromAutoHide();
        }
    }
    return false;
}

void MainWindow::scheduleViewDockAutoHide() {
    if (!_viewAutoHideEnabled || !_viewOverlayPanel) {
        return;
    }
    if (!_viewOverlayPanel->isVisible()) {
        return;
    }
    if (_viewAutoHideTimer) {
        _viewAutoHideTimer->start();
    }
}

void MainWindow::cancelViewDockAutoHide() {
    if (_viewAutoHideTimer) {
        _viewAutoHideTimer->stop();
    }
}

void MainWindow::revealViewDockFromAutoHide() {
    if (!_viewOverlayPanel) {
        return;
    }
    cancelViewDockAutoHide();
    _autoHidePeekTab->hide();
    _viewOverlayPanel->show();
    relayoutCentralOverlays();
}

void MainWindow::applyAutoHideFromUi(bool on) {
    _viewAutoHideEnabled = on;
    cancelViewDockAutoHide();
    syncDockPinButtonState(_viewDockPinButton, on);
    if (_actAutoHidePanel) {
        const QSignalBlocker b(_actAutoHidePanel);
        _actAutoHidePanel->setChecked(on);
    }
    if (!on) {
        _autoHidePeekTab->hide();
        if (_viewOverlayPanel && !_viewOverlayPanel->isVisible()) {
            _viewOverlayPanel->show();
        }
        relayoutCentralOverlays();
    }
}

void MainWindow::attachAndInit() {
    attachQtWindow(_vkWindow, &_vulkanInstance);
    if (!_framePumpTimer) {
        _framePumpTimer = new QTimer(this);
        QObject::connect(_framePumpTimer, &QTimer::timeout, this, &MainWindow::pumpFrame);
    }
    _framePumpTimer->start(16);
}

void MainWindow::pumpFrame() {
    drawFrame();
    syncOrthoSpinsFromAppUnlessFocused();
    updateReadoutLabel(QString::fromStdString(getCursorReadoutText()));
}

void MainWindow::shutdownBeforeQtTeardown() {
    shutdownWindowSystem();
}

MainWindow::ViewState MainWindow::viewStateSnapshot() const {
    ViewState s;
    s.coordinate = _coordinate;
    s.rotation = _rotation;
    s.ortho = _ortho;
    s.window = getWindowSize();
    return s;
}

void MainWindow::applyViewState(const ViewState& s) {
    _coordinate = s.coordinate;
    setRotation(s.rotation);
    _ortho = s.ortho;
    setWindowSize(s.window);
    markOverlayDirty();
}

void MainWindow::openObjFile(const std::string& path) {
    if (!pathLooksLikeObj(path)) {
        std::cout << "Open: not an OBJ file: " << path << '\n';
        return;
    }
    _showCube = false;
    showFlattenedMesh(path);
}

void MainWindow::showCubeMesh() {
    _showCube = true;
    setApplicationVertices(buildCubeVertices());
    rebuildVertexBuffer();
}

void MainWindow::setViewRotation(float degreesX, float degreesY) {
    setRotation(glm::vec2(degreesX, degreesY));
}

void MainWindow::runLscmAndShowMesh() {
    lscm();
    showFlattenedMesh("models/qian_NL.obj");
}

bool MainWindow::captureDepthBufferFloat(std::vector<float>& outDepth, uint32_t& outW, uint32_t& outH) {
    drawFrame();
    return readDepthBufferFloat(outDepth, outW, outH);
}

std::vector<Vertex> MainWindow::buildCubeVertices() const {
    if (!_showCube) {
        return {};
    }

    std::vector<Vertex> v;
    glm::vec3 topColor = glm::vec3(0.0f, 0.0f, 0.5f);
    glm::vec3 bottomColor = glm::vec3(0.0f, 0.0f, 0.4f);
    glm::vec3 frontColor = glm::vec3(0.0f, 0.5f, 0.0f);
    glm::vec3 backColor = glm::vec3(0.0f, 0.4f, 0.0f);
    glm::vec3 leftColor = glm::vec3(0.4f, 0.0f, 0.0f);
    glm::vec3 rightColor = glm::vec3(0.5f, 0.0f, 0.0f);

    auto quad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 color) {
        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        v.push_back(Vertex{ a, n, color });
        v.push_back(Vertex{ b, n, color });
        v.push_back(Vertex{ c, n, color });
        v.push_back(Vertex{ c, n, color });
        v.push_back(Vertex{ d, n, color });
        v.push_back(Vertex{ a, n, color });
    };

    glm::vec3 p000 = { -1.0f, -1.0f, -1.0f };
    glm::vec3 p001 = { -1.0f, -1.0f, 1.0f };
    glm::vec3 p010 = { -1.0f, 1.0f, -1.0f };
    glm::vec3 p011 = { -1.0f, 1.0f, 1.0f };
    glm::vec3 p100 = { 1.0f, -1.0f, -1.0f };
    glm::vec3 p101 = { 1.0f, -1.0f, 1.0f };
    glm::vec3 p110 = { 1.0f, 1.0f, -1.0f };
    glm::vec3 p111 = { 1.0f, 1.0f, 1.0f };

    quad(p000, p001, p101, p100, bottomColor);
    quad(p010, p110, p111, p011, topColor);
    quad(p011, p111, p101, p001, frontColor);
    quad(p010, p000, p100, p110, backColor);
    quad(p010, p011, p001, p000, leftColor);
    quad(p110, p100, p101, p111, rightColor);

    return v;
}

std::vector<Vertex> MainWindow::loadObjTriangles(const std::string& objPath) const {
    struct P3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    const std::filesystem::path inputPath(objPath);
    const std::filesystem::path fullPath = std::filesystem::absolute(inputPath);
    std::cout << "loadObjTriangles() file: " << fullPath.string() << '\n';

    std::ifstream in(fullPath);
    if (!in.is_open()) {
        std::cout << "Failed to open mesh: " << fullPath.string() << '\n';
        return {};
    }

    std::vector<P3> positions;
    std::vector<Vertex> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("v ", 0) == 0) {
            std::istringstream iss(line);
            char c;
            P3 p;
            iss >> c >> p.x >> p.y >> p.z;
            positions.push_back(p);
            continue;
        }
        if (line.rfind("f ", 0) == 0) {
            std::istringstream iss(line);
            char c;
            iss >> c;
            std::vector<int> ids;
            std::string token;
            while (iss >> token) {
                const size_t slash = token.find('/');
                const std::string vStr = (slash == std::string::npos) ? token : token.substr(0, slash);
                if (vStr.empty()) {
                    continue;
                }
                int idx = std::stoi(vStr);
                if (idx > 0) {
                    ids.push_back(idx - 1);
                }
                else if (idx < 0) {
                    ids.push_back(static_cast<int>(positions.size()) + idx);
                }
            }

            if (ids.size() < 3) {
                continue;
            }

            const glm::vec3 n(0.0f, 0.0f, 1.0f);
            const glm::vec3 color(0.2f, 0.8f, 1.0f);
            for (size_t i = 1; i + 1 < ids.size(); ++i) {
                const int i0 = ids[0];
                const int i1 = ids[i];
                const int i2 = ids[i + 1];
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= static_cast<int>(positions.size()) ||
                    i1 >= static_cast<int>(positions.size()) ||
                    i2 >= static_cast<int>(positions.size())) {
                    continue;
                }
                out.push_back(Vertex{ glm::vec3(positions[i0].x, positions[i0].y, positions[i0].z), n, color });
                out.push_back(Vertex{ glm::vec3(positions[i1].x, positions[i1].y, positions[i1].z), n, color });
                out.push_back(Vertex{ glm::vec3(positions[i2].x, positions[i2].y, positions[i2].z), n, color });
            }
        }
    }
    return out;
}

void MainWindow::applyMakeHumanMesh(const makehuman::BodyParameters& params) {
    _mhBodyParams = params;
    _showCube = false;
    const std::vector<makehuman::MeshVertex> mh = makehuman::buildHumanMesh(params);
    if (mh.empty()) {
        std::cout << "applyMakeHumanMesh: empty mesh\n";
        return;
    }
    static_assert(sizeof(Vertex) == sizeof(makehuman::MeshVertex), "Vertex layout must match makehuman::MeshVertex");
    std::vector<Vertex> mesh(mh.size());
    std::memcpy(mesh.data(), mh.data(), mh.size() * sizeof(Vertex));
    setApplicationVertices(mesh);
    rebuildVertexBuffer();
    std::cout << "MakeHuman mesh: " << (mesh.size() / 3) << " triangles (base: " << makehuman::resolveBaseObjPath() << ")\n";
}

void MainWindow::showFlattenedMesh(const std::string& objPath) {
    std::vector<Vertex> mesh = loadObjTriangles(objPath);
    if (mesh.empty()) {
        std::cout << "No triangles loaded from " << objPath << '\n';
        return;
    }

    glm::vec3 minP(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 maxP(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const Vertex& v : mesh) {
        minP = glm::min(minP, v.pos);
        maxP = glm::max(maxP, v.pos);
    }
    const glm::vec3 center = (minP + maxP) * 0.5f;

    for (auto& v : mesh) {
        v.pos -= center;
    }

    setApplicationVertices(mesh);
    rebuildVertexBuffer();
    std::cout << "Rendered flattened mesh: " << objPath << " (" << mesh.size() / 3 << " triangles)\n";
}
