def add_fev1_prct_pred_line(fig, val, y_max, row, col, width=2):
    fig.add_shape(
        type="line",
        x0=val,
        y0=0,
        x1=val,
        y1=y_max * 1.1,
        line=dict(color="black", width=width, dash="dash"),
        # fillcolor="red",
        # line_width=0,
        row=row,
        col=col,
    )
    return -1
