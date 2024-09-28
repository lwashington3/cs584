import numpy as np


def main(data:list[np.ndarray], labels:np.ndarray, theta:np.ndarray, eta:float):
	next_theta = []
	with open("training_phase.tex", 'w') as f:
		for i, xi in enumerate(data):
			f.write("\\begin{table}[H]\n"
					"\t\\centering\n"
					"\t\\caption{}\n"
					f"\t\\label{{tab:training-{i}}}\n"
					"\t\\begin{tabular}{*{4}{|c}|}\n"
					"\t\t\\hline\n"
					f"\t\t\\textcolor{{lightblue}}{{$x$}} & \\textcolor{{red}}{{$y$}} & $\\theta^{{T}}\\vec{{x}}$ & $\\left( \\frac{{1}}{{1 + e^{{-\\vec{{\\theta}}^{{T}}\\vec{{x}}}} }} - \\textcolor{{red}}{{y}} \\right) \\textcolor{{lightblue}}{{x_{{{i}}}}}$\\\\\n")
			values = []
			for x, y in zip(data, labels):
				x_str = f"[{', '.join(map(str, x))}]"
				theta_str = f"[{', '.join(map(str, theta))}]"
				theta_x = theta@x
				theta_x_str = f"{theta_x:,}"
				neg_theta_x_str = f"{-theta_x:,}"
				value = 1 / (1 + np.exp(theta_x))
				value -= y
				value *= x[i]
				value = value if value != 0 else 0  # Some of the zeros had a negative attached to them
				value_str = f"{value:,}"
				values.append(value)

				f.write("\t\t\\hline\n"
						f"\t\t{x_str} & \\textbf{{{y}}} & ${theta_str} \\times {x_str} = {theta_x_str}$ & $\\left( \\frac{{1}}{{1 + e^{{{neg_theta_x_str}}}}} - \\textcolor{{red}}{{\\mathbf{{{y}}}}} \\right) \\times \\textcolor{{lightblue}}{{{x[i]}}} = {value_str}$\\\\\n")
			total = sum(values)
			f.write("\t\t\\hline\n"
					"\t\\end{tabular}\n"
					"\\end{table}\n"
					"\\begin{equation*}\n"
					"\\begin{aligned}[eqpurple]\n"
					f"\t\\sum_{{i=1}}^{len(data[0])} x_{{i{i}}} \\left[ \\frac{{1}}{{ 1 + e^{{\\vec{{\\theta^{{T}}}}\\vec{{x}}}}}} - y_{{i}} \\right] &= {' + '.join(map(str, values)).replace('+ -', '- ')} \\\\\n"
					f"&= {total :,}\n"
					"\\end{aligned}\n"
					"\\end{equation*}\n")
			next_theta.append(total)

	theta = theta - ((eta / len(data)) * np.array(next_theta))

	with open("testing_phase.tex", 'w') as f:
		f.write("\\begin{table}[H]\n"
				"\t\\centering\n"
				"\t\\caption{}\n"
				f"\t\\label{{tab:training-{i}}}\n"
				"\t\\begin{tabular}{*{5}{|c}|}\n"
				"\t\t\\hline\n"
				f"\t\t\\textcolor{{lightblue}}{{$x$}} & \\textcolor{{red}}{{$y$}} & $\\theta^{{T}}\\vec{{x}}$ & $\\mathbf{{P}} = \\left( \\frac{{1}}{{1 + e^{{-\\vec{{\\theta}}^{{T}}\\vec{{x}}}} }} - y \\right) $ & \\textcolor{{red}}{{\\textbf{{Predicted Class}} $\\hat{{y}}$}}\\\\\n")

		for x, y in zip(data, labels):
			x_str = f"[{', '.join(map(str, x))}]"
			theta_str = f"[{', '.join(map(lambda value: str(value).replace('.0', ''), theta))}]"
			theta_x = theta@x
			theta_x_str = f"{theta_x:,}"
			p = 1 / (1 + np.exp(-theta_x))

			f.write("\t\t\\hline\n"
					f"\t\t{x_str} & \\textbf{{{y}}} & ${theta_str} \\times {x_str} = {theta_x_str}$ & {p:.9f} & \\textbf{{{1 if p >= 0.5 else 0}}} \\\\\n")
		f.write("\t\t\\hline\n"
				"\t\\end{tabular}\n"
				"\\end{table}\n")


if __name__ == "__main__":
	data = [
		np.array([1, 1, 1, 0, 1, 1]),
		np.array([1, 0, 0, 1, 1, 0]),
		np.array([1, 0, 1, 1, 0, 0]),
		np.array([1, 1, 0, 0, 1, 0]),
		np.array([1, 1, 0, 1, 0, 1]),
		np.array([1, 1, 0, 1, 1, 0])
	]

	labels = np.array([1, 0, 1, 0, 1, 0])

	theta = np.zeros(shape=labels.shape, dtype=np.int8)
	main(data, labels, theta, 3)
