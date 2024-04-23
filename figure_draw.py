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
dedup160m_32_96 = [34570570, 98217163, 8730153, 2465050, 789394, 418952, 306127, 187018, 176703, 146568, 173166, 251136]


dedup410m_32_48 = [55667725, 36772784,31082769,6294070,6398771,1818303,2477312,1673641,689333,1208393,585278,1763621]
dedup410m_32_64 = [45892775, 73023639, 15468711, 4973983, 2340836, 1276892, 994909, 494626, 387584, 362349, 413059, 802637]
dedup410m_32_96 = [32413858, 98468010, 9689734, 2852055, 938247, 506305, 371739, 226300, 212996, 174814, 216367, 361575]

dedup1b_32_48 = [53337373, 36207822, 31780490, 6668892, 6902386, 1978310, 2706078, 1840521, 765859, 1365831, 680658, 2197780]
dedup1b_32_64 = [44012189, 72674249, 16309375, 5372333, 2566720, 1428121, 1125248, 564809, 447231, 419171, 493153, 1019401]

dedup2_8b_32_48 = [50521557, 35429015, 32545886, 7126382, 7536738, 2186161, 3005835, 2062528, 862821, 1561455, 805117, 2788505]

dedup6_9b_32_48 = [49156099, 34945652, 32775559, 7327698, 7835542, 2292935, 3156483, 2172148, 912410, 1670842, 886455, 3300177]

dedup12b_32_48 = [48241477, 34621975, 32916929, 7468693, 8047337, 2368123, 3265290, 2253386, 949157, 1745443, 939923, 3614267]





memorized_32_16 = [894809, 1309026, 1763621, 2197780, 2788505, 3300177, 3614267]
unmemorized_32_16 = [64506233, 59464820, 55667725, 53337373, 50521557, 48241477]
half_memorized_32_16 = [1321896, 1584734, 1818303, 1978310, 2186161, 2292935,2368123]
plt.plot(["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"], memorized_32_16, label="Memorized Sentence", color="blue",
         marker="o", linestyle="-", linewidth=2)
plt.plot(["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"], unmemorized_32_16, label="Unmemorized Sentence", color="red",
         marker="o", linestyle="-", linewidth=2)
plt.plot(["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"], half_memorized_32_16, label="Half Memorized Sentence",
         color="green", marker="o", linestyle="-", linewidth=2)
plt.xlabel("Model Size", fontsize=14)
plt.ylabel("Number of Memorized Sentences", fontsize=14)
plt.title("Number of Sentences Memorized by Model Size", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("memorized.png")
plt.show()

memoirzed_70m_dynamic_complement = [894809, 411428, 262729, 167846, 90968]
memoirzed_160m_dynamic_complement = [1309026, 583993, 251136]
memoirzed_410m_dynamic_complement = [1763621, 802637, 361575]
plt.plot(["16", "32", "64", "96"],memoirzed_70m_dynamic_complement, label="70m", color="blue", marker="o", linestyle="-", linewidth=2)
plt.plot(["16", "32", "64", "96"],memoirzed_160m_dynamic_complement, label="160m", color="red", marker="o", linestyle="-", linewidth=2)
plt.xlabel("Complement Size", fontsize=14)
plt.ylabel("Number of Memorized Sentences", fontsize=14)
plt.title("Number of Sentences Memorized vs Complement Size", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("memorized_dynamic_complement.png")
plt.show()

memorized_70m_dynamic_context = [1273552, 1591431, 2079244, 2079244]
memorized_160m_dynamic_context = [1309026]
memorized_410m_dynamic_context = [1763621]
memorized_1b_dynamic_context = [2197780]
memorized_2_8b_dynamic_context = [2788505]
memorized_6_9b_dynamic_context = [3300177]
memorized_12b_dynamic_context = [3614267]

plt.plot(["32", "48", "64", "96"],memorized_70m_dynamic_context, label="70m", color="blue", marker="o", linestyle="-", linewidth=2)
plt.plot(["32"],memorized_160m_dynamic_context, label="160m", color="red", marker="o", linestyle="-", linewidth=2)
plt.plot(["32"],memorized_410m_dynamic_context, label="410m", color="green", marker="o", linestyle="-", linewidth=2)
plt.plot(["32"],memorized_1b_dynamic_context, label="1b", color="purple", marker="o", linestyle="-", linewidth=2)
plt.plot(["32"],memorized_2_8b_dynamic_context, label="2.8b", color="orange", marker="o", linestyle="-", linewidth=2)
plt.plot(["32"],memorized_6_9b_dynamic_context, label="6.9b", color="brown", marker="o", linestyle="-", linewidth=2)
plt.plot(["32"],memorized_12b_dynamic_context, label="12b", color="black", marker="o", linestyle="-", linewidth=2)
plt.xlabel("Context Size", fontsize=14)
plt.ylabel("Number of Memorized Sentences", fontsize=14)
plt.title("Number of Sentences Memorized vs Context Size", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("memorized_dynamic_context.png")
plt.show()


