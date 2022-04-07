from PyQt5.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView, QDialog, QFileSystemModel, QListWidget


class MultiDirSelector(QFileDialog):
    def __init__(self, start_dir, *args):
        super(MultiDirSelector, self).__init__(*args)
        self.setDirectory(start_dir)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.DirectoryOnly)
        self.list = []
        self.current_dir = None
        for view in self.findChildren((QListView, QTreeView)):
            if isinstance(view.model(), QFileSystemModel):
                view.setSelectionMode(QAbstractItemView.ExtendedSelection)
                self.current_dir = view.model().rootDirectory().absolutePath()
        if self.exec_() == QDialog.Accepted:
            self.list += self.selectedFiles()
        if self.current_dir and self.list == [self.current_dir]:
            self.list = []
        self.deleteLater()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ex = MultiDirSelector()
    print(ex.list)
    sys.exit(app.exec_())
