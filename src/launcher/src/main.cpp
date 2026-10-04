#include "mainDialog.h"
#include "launcherTheme.h"

#include "common/threads.h"

#include <QApplication>

int main(int argc, char* argv[]) {
	// Lock the launcher (and, via CreateProcess inheritance, every emulator
	// child it spawns) to the P-core mask before any Qt thread starts.
	Common::InitializeThreads();
	QApplication a(argc, argv);
	LauncherTheme::Initialize(a);

	MainDialog w;

	w.emit Start();

	w.show();

	return QApplication::exec();
}
