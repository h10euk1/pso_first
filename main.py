import numpy as np

def mean_square(state):
    """ mean_squareは状態の二乗平均を計算する

        引数:
            state: 評価される状態
    """
    return (state ** 2).mean() # mean-squared

class Particle:
    I = 0.8  # 慣性係数
    Ag = 0.9 # gbestの加速係数
    Ap = 0.9 # pbestの加速係数
    def __init__(self, n, xmin, xman, vmin, vmax, eval_func=mean_square):
        """ Particleは粒子を表す

            引数:
                n: 状態の次元
                xmin: 状態の初期値の最小値
                xmax: 状態の初期値の最大値
                vmin: 速度の初期値の最小値
                vmax: 速度の初期値の最大値
                eval_func: 状態の評価関数
        """
        # 状態、速度の初期値を設定する（一様乱数）
        self.state, self.velocity = self._rand_state_velocity(n, xmin, xman, vmin, vmax)
        # 評価関数を引数で与えられたものに設定する
        self._evaluate = eval_func
        # 初期状態とそのときのスコアを暫定ベストにする
        self.best = {"state": self.state.copy(),
                     "score": self._evaluate(self.state)}

    def update(self, gb_state):
        """ updateは粒子の状態を更新する。
            大谷紀子『進化計算アルゴリズム入門』、p. 112の式を利用。

            引数:
                gb_state: グローバルベストの状態
        """
        # パーソナルベストを利用した速度の変化量
        delta_vel_by_p = np.random.random() * (self.best["state"] - self.state)
        # グローバルベストを利用した速度の変化量
        delta_vel_by_g = np.random.random() * (gb_state - self.state)
        # 速度を更新
        self.velocity = self.I * self.velocity + self.Ap * delta_vel_by_p \
                                               + self.Ag * delta_vel_by_g
        # 状態を更新
        self.state = self.state + self.velocity
        # 更新した状態を評価してスコアを導出
        new_score = self._evaluate(self.state)
        # 新しいスコアが以前のものより小さかったらベストを更新
        if new_score < self.best["score"]:
            self.best["state"] = self.state.copy()
            self.best["score"] = new_score

    def _rand_state_velocity(self, size, xmin, xmax, vmin, vmax):
        """ _rand_state_velocityはランダムな状態と速度を生成する

            引数:
                size: 状態の次元
                xmin: 状態の初期値の最小値
                xmax: 状態の初期値の最大値
                vmin: 速度の初期値の最小値
                vmax: 速度の初期値の最大値
        """
        state = np.random.uniform(xmin, xmax, size)
        velocity = np.random.uniform(vmin, vmax, size)
        return state, velocity

def particle_test():
    # 粒子の初期化がうまく行くかのテスト
    p = Particle(2, 0, 1, 0, 1)
    assert (p.state == p.best["state"]).all()
    assert (np.mean(p.state ** 2) == p.best["score"]).all()

    # 状態の更新がうまく行くかのテスト
    p = Particle(2, 0, 1, 0, 1)
    virtual_global_best = np.zeros(2)
    for _ in range(1000):
        p.update(virtual_global_best)
    assert((p.best["state"] @ p.best["state"]) < 1E-3)

    # 評価関数を引数で与えたときのテスト
    p = Particle(2, 0, 1, 0, 1, eval_func=lambda s: np.sum(s))
    assert(p._evaluate(p.state) == np.sum(p.state))

    print("All the tests were passed!")

if __name__ == "__main__":
    particle_test()
