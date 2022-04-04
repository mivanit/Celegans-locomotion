@staticmethod
def pos_multi_conns(
        # search in this directory
        rootdir: Path,
        conn_key: str,
        *args,
        # args passed down to `_draw_setup.py()`
        bodydat: Path = Path('body.dat'),
        collobjs: Path = Path('coll_objs.tsv'),
        food_location: List[str] = ["left", "right", "none"],
        params: Optional[Path] = Path('params.json'),
        time_window: Tuple[OptInt, OptInt] = (None, None),
        figsize_scalar: Optional[float] = None,
        pad_frac: Optional[float] = None,
        # args specific to this plotter
        idx: int = 0,
        show: bool = True,
        only_final: bool = False,
):
    folder_list = []
    for folder in os.listdir(rootdir):
        print(folder, type(folder))
        if conn_key in folder:  # and os.path.isdir(folder):
            folder_list.append(os.path.join(rootdir, folder))

    if folder_list:
        if not isinstance(folder_list[0], Path):
            new_rootdir = Path(folder_list[0])

        pdbg(folder_list[0])
        pdbg(bodydat)
        pdbg(folder_list[0] + '/**/' + bodydat)
        default_dir = folder_list[0] + '/food_' + food_location[0] + '/'
        print(f'> using as default: {default_dir}')

        fig, ax, data_default, bounds = _draw_setup(
            rootdir=default_dir,
            bodydat=bodydat,
            collobjs=collobjs,
            # params = params,
            time_window=time_window,
            figsize_scalar=figsize_scalar,
            pad_frac=figsize_scalar,
        )

        for conn_dir in folder_list:
            for food in food_location:
                x_dir = conn_dir + '/food_' + food
                x_bodydat: str = joinPath(x_dir, bodydat)
                x_params: str = joinPath(x_dir, params)

                data: NDArray[(int, int), CoordsRotArr] = read_body_data(x_bodydat)

                head_data: NDArray[Any, CoordsRotArr] = data[-1, idx]
                if not only_final:
                    head_data = data[:, idx]

                print(x_bodydat)
                print(head_data.shape, head_data.dtype)

                if only_final:
                    ax.plot(head_data['x'], head_data['y'], 'o', label=x_dir)
                else:
                    ax.plot(head_data['x'], head_data['y'], label=x_dir)
        # tup_foodpos = _plot_foodPos(ax, x_params, label = x_dir)
        # print(tup_foodpos)

        ax.set_title(rootdir + '/**/')
        plt.legend()

        if show:
            plt.show()
    else:
        raise FileNotFoundError('Could not find any matching files')

