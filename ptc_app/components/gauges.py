"""Error gauge component — visible colored precision indicator."""
import streamlit as st


def error_gauge(label: str, calc: float, exp: float,
                unit: str = "", fmt: str = ".3f"):
    """Render a metric with colored precision indicator.

    Green < 1%, Yellow 1-5%, Red > 5%.
    Shows: label, calc value, exp value, error %, colored status box.
    If exp is None or 0, just show the value.
    """
    if exp is None or exp == 0:
        st.markdown(f"**{label}**: `{calc:{fmt}}` {unit}")
        return

    err_pct = abs((calc - exp) / exp * 100)
    signed_err = (calc - exp) / exp * 100

    if err_pct < 1.0:
        bg_color = "#d4edda"   # light green
        border_color = "#28a745"
        text_color = "#155724"
        status = "< 1%"
    elif err_pct < 5.0:
        bg_color = "#fff3cd"   # light yellow
        border_color = "#ffc107"
        text_color = "#856404"
        status = f"{err_pct:.1f}%"
    else:
        bg_color = "#f8d7da"   # light red
        border_color = "#dc3545"
        text_color = "#721c24"
        status = f"{err_pct:.1f}%"

    st.markdown(
        f"""<div style="
            background:{bg_color};
            border-left:4px solid {border_color};
            border-radius:4px;
            padding:8px 12px;
            margin-bottom:10px;
        ">
            <div style="font-weight:600;font-size:0.9em;color:#333;margin-bottom:2px;">
                {label}
            </div>
            <div style="font-size:1.1em;font-weight:700;color:{text_color};">
                {calc:{fmt}} {unit}
            </div>
            <div style="font-size:0.8em;color:#666;margin-top:2px;">
                exp: {exp:{fmt}} {unit} &nbsp;|&nbsp;
                err: {signed_err:+.2f}% &nbsp;
                <span style="
                    background:{border_color};
                    color:white;
                    padding:1px 6px;
                    border-radius:3px;
                    font-weight:600;
                    font-size:0.85em;
                ">{status}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def gauge_row(metrics: list):
    """Render multiple gauges in columns.
    metrics: list of (label, calc, exp, unit, fmt) tuples.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            label, calc, exp = m[0], m[1], m[2]
            unit = m[3] if len(m) > 3 else ""
            fmt = m[4] if len(m) > 4 else ".3f"
            error_gauge(label, calc, exp, unit, fmt)
