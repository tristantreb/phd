def add_o2sat_normal_range_line(fig, y_max, row, col):
    fig.add_shape(
        type="line",
        # opacity=0.1,
        x0=94,
        y0=0,
        x1=94,
        y1=y_max*1.1,
        line=dict(color="red", width=1, dash="dash"),
        # fillcolor="red",
        # line_width=0,
        row=row,
        col=col,
    )
    return -1
