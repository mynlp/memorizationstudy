from matplotlib import pyplot as plt


range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
dedup70m_32_48 = [64506233,37841243,27559121,4910421,4754192,1321896,1784455,1179768,476356,823418,380088,894809]
dedup70m_32_64 = [52919170, 72557211,12500242,3751401,1689877,874961,657503,324606,253022,242386,250193,411428]
dedup70m_32_80 = [44099661,84685911,12138802,2763596,1021318,414016,321856,221454,177818,175597,149242,262729]
dedup70m_32_96 = [37165380,97495916,7774461,2070021,634902,330683,242755,151336,144813,122512,131375,167846]
dedup70m_32_112 = [27089493,111089211,5890283,1064150,453601,220630,164916,124705,87874,82222,73947,90968]

dedup70m_48_64 = [61979944, 37252567, 28109245, 5234289, 5266252, 1511858, 2096874, 1447860, 604312, 1090535, 564712, 1273552]
dedup70m_64_80 = [60512868, 36851547, 28355726, 5411213, 5551329, 1623974, 2283239, 1613176, 684451, 1266992, 686054, 1591431]
dedup70m_64_96 = [58838534, 36359295, 28572092, 5605379, 5872569, 1747742, 2491068, 1797351, 776486, 1464090, 828150, 2079244]
dedup70m_96_112 = [58838534, 36359295, 28572092, 5605379, 5872569, 1747742, 2491068, 1797351, 776486, 1464090, 828150, 2079244]


dedup160m_32_48 = [59464820,37414987,29680540,5670919,5630038,1584734,2147099,1439956,589894,1018286,481701,1309026]
dedup160m_32_64 = [48981644,73082309,14089895,4388330,2025764,1080489,832277,413358,321405,300862,331674,583993]
dedup160m_32_80 = [40945267, 85106596, 13631465, 3257000, 1259311, 525821, 411514, 281785, 223058, 213618, 201005, 375560]
dedup160m_32_96 = [34570570, 98217163, 8730153, 2465050, 789394, 418952, 306127, 187018, 176703, 146568, 173166, 251136]


dedup410m_32_48 = [55667725, 36772784,31082769,6294070,6398771,1818303,2477312,1673641,689333,1208393,585278,1763621]
dedup410m_32_64 = [45892775, 73023639, 15468711, 4973983, 2340836, 1276892, 994909, 494626, 387584, 362349, 413059, 802637]
dedup410m_32_80 = [38387191, 85029583, 15027621, 3727957, 1487785, 631148, 498548, 340818, 270055, 256867, 252660, 521767]
dedup410m_32_96 = [32413858, 98468010, 9689734, 2852055, 938247, 506305, 371739, 226300, 212996, 174814, 216367, 361575]

dedup1b_32_48 = [53337373, 36207822, 31780490, 6668892, 6902386, 1978310, 2706078, 1840521, 765859, 1365831, 680658, 2197780]
dedup1b_32_64 = [44012189, 72674249, 16309375, 5372333, 2566720, 1428121, 1125248, 564809, 447231, 419171, 493153, 1019401]
dedup1b_32_80 = [36867017, 84647974, 15912789, 4065149, 1665085, 721123, 574521, 395815, 311601, 296946, 306808, 667172]
dedup1b_32_96 = [31173743, 98250111, 10330382, 3143473, 1069186, 584537, 430670, 264792, 247764, 200727, 262264, 474351]

dedup2_8b_32_48 = [50521557, 35429015, 32545886, 7126382, 7536738, 2186161, 3005835, 2062528, 862821, 1561455, 805117, 2788505]


dedup6_9b_32_48 = [49156099, 34945652, 32775559, 7327698, 7835542, 2292935, 3156483, 2172148, 912410, 1670842, 886455, 3300177]

dedup12b_32_48 = [48241477, 34621975, 32916929, 7468693, 8047337, 2368123, 3265290, 2253386, 949157, 1745443, 939923, 3614267]






memorized_32_16 = [894809, 1309026, 1763621, 2197780, 2788505, 3300177, 3614267]
unmemorized_32_16 = [64506233, 59464820, 55667725, 53337373, 50521557, 49156099, 48241477]
memorized_0_1_32_16 = [37841243, 37414987, 36772784, 36207822, 35429015, 34945652, 34621975]
memorized_0_2_32_16 = [27559121, 29680540, 31082769, 31780490, 32545886, 32775559, 32916929]
memorized_0_3_32_16 = [4910421, 5670919, 6294070, 6668892, 7126382, 7327698, 7468693]
memorized_0_4_32_16 = [4754192, 5630038, 6398771, 6902386, 7536738, 7835542, 8047337]
memorized_0_5_32_16 = [1321896, 1584734, 1818303, 1978310, 2186161, 2292935, 2368123]
memorized_0_6_32_16 = [1784455, 2147099, 2477312, 2706078, 3005835, 3156483, 3265290]
memorized_0_7_32_16 = [1179768, 1439956, 1673641, 1840521, 2062528, 2172148, 2253386]
memorized_0_8_32_16 = [476356, 589894, 689333, 765859, 862821, 912410, 949157]
memorized_0_9_32_16 = [823418, 1018286, 1208393, 1365831, 1561455, 1670842, 1745443]
model_size = ["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]

# Create a list to store all data and labels for easy iteration
data = [
    (unmemorized_32_16, "Memorize Score 0", "red", "v", "--"),
    (memorized_0_1_32_16, "Memorize Score 0.1", "green", "s", ":"),
    (memorized_0_2_32_16, "Memorize Score 0.2", "purple", "^", "-."),
    (memorized_0_3_32_16, "Memorize Score 0.3", "orange", "p", "-"),
    (memorized_0_4_32_16, "Memorize Score 0.4", "pink", "*", "--"),
    (memorized_0_5_32_16, "Memorize Score 0.5", "cyan", "H", ":"),
    (memorized_0_6_32_16, "Memorize Score 0.6", "olive", "+", "-."),
    (memorized_0_7_32_16, "Memorize Score 0.7", "darkgreen", "D", "-"),
    (memorized_0_8_32_16, "Memorize Score 0.8", "yellow", "X", "--"),
    (memorized_0_9_32_16, "Memorize Score 0.9", "black", ".", ":"),
    (memorized_32_16, "Memorize Score 1", "blue", "o", "-")
]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))# Plot each data series
for info in data:
    array, label, color, marker, linestyle = info
    axs[0].plot(model_size, array, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=2, alpha=0.7)
# plt.yscale('log')
# plt.xlabel("Model Size", fontsize=14)
# plt.ylabel("Number of Memorized Sentences", fontsize=14)
# plt.title("Number of Sentences Memorized by Model Size", fontsize=16)
# plt.legend(fontsize=10, loc='upper left')
# plt.grid(True)
# plt.savefig("memorized.png", bbox_inches='tight', dpi=600)
# plt.show()
axs[0].set_yscale('log')
axs[0].set_xlabel("Model Size", fontsize=14)
axs[0].set_ylabel("Number of Sentences", fontsize=14)
axs[0].set_title("Sentences of Different Memorization Score vs Model Size", fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(fontsize=10, loc='upper left')
fig.savefig('memorized_sentences.png', bbox_inches='tight', dpi=600)

x_labels = ["32", "48", "64", "96"]
memorized_70m_dynamic_complement = [894809, 411428, 262729, 167846]
memorized_160m_dynamic_complement = [1309026, 583993, 375560, 251136]
memorized_410m_dynamic_complement = [1763621, 802637, 521767, 361575]
memorized_1b_dynamic_complement = [2197780, 1019401, 667172, 474351]
memorized_2_8b_dynamic_complement = [2788505, 1297351, 877233, 631442]
memorized_6_9b_dynamic_complement = [3300177, 1584084, 1106232, 818611]
memorized_12b_dynamic_complement = [3614267, 1732833, 1247639, 935722]
SUBSETS = ['70m', '160m', '410m', '1b', '2_8b', '6_9b', '12b']
COLOURS = ["blue", "red", "green", "purple", "orange", "brown", "black"]
MARKERS = ["o", "v", "s", "p", "*", "+", "x"]
for subset, colour, marker in zip(SUBSETS, COLOURS, MARKERS):
    axs[1].plot(x_labels, eval(f"memorized_{subset}_dynamic_complement"), label=subset, color=colour, marker=marker, linestyle=":", linewidth=2)
# plt.plot(x_labels, memoirzed_70m_dynamic_complement, label="70m", color="blue", marker="o", linestyle=":", linewidth=2)
# plt.plot(x_labels, memoirzed_160m_dynamic_complement, label="160m", color="red", marker="v", linestyle=":", linewidth=2)
# plt.plot(x_labels, memoirzed_410m_dynamic_complement, label="410m", color="green", marker="s", linestyle="-.", linewidth=2)
# plt.plot(x_labels, memoirzed_1b_dynamic_complement, label="1b", color="purple", marker="p", linestyle="--", linewidth=2)
# plt.plot(x_labels, memoirzed_2_8b_dynamic_complement, label="2.8b", color="orange", marker="*", linestyle="--", linewidth=2)
# plt.plot(x_labels, memoirzed_6_9b_dynamic_complement, label="6.9b", color="brown", marker="+", linestyle="-.", linewidth=2)
# plt.plot(x_labels, memoirzed_12b_dynamic_complement, label="12b", color="black", marker="x", linestyle=":", linewidth=2)
# plt.xlabel("Complement Size", fontsize=14)
# plt.ylabel("Number of Memorized Sentences", fontsize=14)
# plt.title("Number of Sentences Memorized vs Complement Size", fontsize=16)
# plt.legend(title='Model Sizes:', title_fontsize='10', fontsize='10', loc='upper left')
axs[1].set_xlabel("(b) Complement Size", fontsize=14)
axs[1].set_ylabel("Number of Memorized Sentences (Millions)", fontsize=14)
axs[1].set_title("Number of Sentences Memorized vs Complement Size", fontsize=14)
axs[1].legend(title='(a) Model Sizes:', title_fontsize='10', fontsize='10', loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.5)
fig.savefig('memorized_dynamic_complement.png', bbox_inches='tight', dpi=600)

# plt.savefig("memorized_dynamic_context.png", bbox_inches='tight', dpi=600)
# plt.show()

memorized_70m_dynamic_context = [1273552, 1591431, 2079244, 3105332]
memorized_160m_dynamic_context = [1309026, 1675521, 2204431, 3388421]
memorized_410m_dynamic_context = [1763621, 2291931, 3094105, 4641115]
memorized_1b_dynamic_context = [2197780, 2923047, 4033804,  6131382]
memorized_2_8b_dynamic_context = [2788505, 3848136,5348909, 8023383 ]
memorized_6_9b_dynamic_context = [3300177, 4648803, 6415035,  9611496]
memorized_12b_dynamic_context = [3614267, 5096116, 7033750, 10523472]
for subset, colour, marker in zip(SUBSETS, COLOURS, MARKERS):
    axs[2].plot(x_labels, eval(f"memorized_{subset}_dynamic_context"), label=subset, color=colour, marker=marker, linestyle=":", linewidth=2)
# plt.plot(x_labels, memorized_70m_dynamic_context, label="70m", color="blue", marker="o", linestyle=":", linewidth=2)
# plt.plot(x_labels, memorized_160m_dynamic_context, label="160m", color="red", marker="v", linestyle=":", linewidth=2)
# plt.plot(x_labels, memorized_410m_dynamic_context, label="410m", color="green", marker="s", linestyle="-.", linewidth=2)
# plt.plot(x_labels, memorized_1b_dynamic_context, label="1b", color="purple", marker="p", linestyle="--", linewidth=2)
# plt.plot(x_labels, memorized_2_8b_dynamic_context, label="2.8b", color="orange", marker="*", linestyle="--", linewidth=2)
# plt.plot(x_labels, memorized_6_9b_dynamic_context, label="6.9b", color="brown", marker="+", linestyle="-.", linewidth=2)
# plt.plot(x_labels, memorized_12b_dynamic_context, label="12b", color="black", marker="x", linestyle=":", linewidth=2)
axs[2].set_xlabel("(c) Context Size", fontsize=14)
axs[2].set_ylabel("Number of Memorized Sentences (Millions)", fontsize=14)
axs[2].set_title("Number of Sentences Memorized vs Context Size", fontsize=14)
axs[2].legend(title='Model Sizes:', title_fontsize='10', fontsize='10', loc='upper left')
axs[2].grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("Context Size", fontsize=14)
# plt.ylabel("Number of Memorized Sentences", fontsize=14)
# plt.title("Number of Sentences Memorized vs Context Size", fontsize=16)
# plt.legend(title='Model Sizes:', title_fontsize='10', fontsize='10', loc='upper left')
# plt.grid(True, linestyle='--', alpha=0.5)
fig.savefig('memorized_dynamic_context.png', bbox_inches='tight', dpi=600)

plt.tight_layout()  # adjust subplot params to give specified padding
plt.savefig("memorized.png", bbox_inches='tight', dpi=600)
plt.show()


