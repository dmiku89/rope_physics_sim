from rope_animation import animated_rope_plot
from rope_model import Rope

if __name__ == '__main__':
    """Пример расчета троса, с демонстрвцией анимации в Matplotlib"""
    rope = Rope(init_angle=120.0, n_chains=10, G_cargo=20.0, k=15000.0, k_damp=30.0)
    rope.calculate(60.0, dt=0.001, V0=1.0, record_step=50)

    animated_rope_plot(rope)