#!/usr/bin/env python3
"""
Train a BPE tokenizer on InChI strings for downstream embedding models.
Input file: one InChI per line.
Output: a tokenizer saved in the Hugging Face format.
"""

import os
import sys
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the InChI data file (assumed to be in the same directory)
    inchi_file = os.path.join(script_dir, "inchi_output.txt")
    
    # Check that the file exists
    if not os.path.isfile(inchi_file):
        print(f"ERROR: InChI file not found at:\n{inchi_file}")
        print("Please ensure 'inchi_output.txt' is in the same folder as this script.")
        sys.exit(1)
    
    # Directory where the trained tokenizer will be saved (subfolder of script dir)
    tokenizer_dir = os.path.join(script_dir, "inchi_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Initialize a byte-level BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Special tokens commonly used for transformer models (like RoBERTa)
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    # Train the tokenizer on the InChI file
    # Adjust vocab_size and min_frequency based on your data
    tokenizer.train(
        files=[inchi_file],
        vocab_size=2000,                # Small vocab is enough for chemistry tokens
        min_frequency=2,                 # Ignore tokens that appear only once
        special_tokens=special_tokens
    )

    # Set the post-processor to handle special tokens correctly (RoBERTa style)
    tokenizer.post_processor = BertProcessing(
        sep=("</s>", tokenizer.token_to_id("</s>")),
        cls=("<s>", tokenizer.token_to_id("<s>"))
    )

    # Enable truncation and padding if you plan to use it directly
    # set default to avoid having to handle it in the trainer/dataloader
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    # Save the tokenizer to disk
    tokenizer.save_model(tokenizer_dir)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    print(f"Tokenizer trained and saved to {tokenizer_dir}")

    # Quick test: tokenize a sample InChI
    test_inchi = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    encoded = tokenizer.encode(test_inchi)
    print("\nTest tokenization:")
    print(f"Input: {test_inchi}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")

if __name__ == "__main__":
    #main() uncomment this line to run the training when executing the script

    # Quick test: tokenize a sample InChI
    from tokenizers import ByteLevelBPETokenizer
        
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    vocab_path = os.path.join(script_dir, "inchi_tokenizer", "vocab.json")
    merges_path = os.path.join(script_dir, "inchi_tokenizer", "merges.txt")

    tokenizer_bpe = ByteLevelBPETokenizer(vocab=vocab_path, merges=merges_path)
    print("Loaded ByteLevelBPETokenizer")
        
    # Example usage
    encoded = tokenizer_bpe.encode("InChI=1S/C325H387N118O209P29/c1-101-40-418(109(9)374-253(101)468)159-33-119(624-654(504,505)572-58-139-214(192(453)283(608-139)430-86-360-170-239(430)379-299(334)393-261(170)476)644-673(542,543)580-65-143-212(190(451)281(607-143)428-84-358-168-233(332)350-80-354-237(168)428)642-669(534,535)576-61-135-208(185(446)276(602-135)415-28-24-156(327)376-314(415)492)632-659(514,515)569-55-132-125(39-165(599-132)424-46-107(7)259(474)412-322(424)500)630-681(558,559)591-77-324-229(467)228(561-100-562-230(324)110-15-11-10-12-16-110)297(622-324)425-47-108(8)260(475)413-323(425)501)126(593-159)50-564-660(516,517)636-218-146(613-287(196(218)457)434-90-364-174-243(434)383-303(338)397-265(174)480)69-585-676(548,549)649-226-153(620-295(204(226)465)442-98-372-182-251(442)391-311(346)405-273(182)488)75-590-680(556,557)651-225-152(619-294(203(225)464)441-97-371-181-250(441)390-310(345)404-272(181)487)73-587-675(546,547)646-215-140(609-284(193(215)454)431-87-361-171-240(431)380-300(335)394-262(171)477)59-573-657(510,511)626-120-34-160(419-41-102(2)254(469)407-317(419)495)594-127(120)49-563-653(502,503)625-121-35-161(420-42-103(3)255(470)408-318(420)496)595-128(121)51-565-661(518,519)639-221-148(615-290(199(221)460)437-93-367-177-246(437)386-306(341)400-268(177)483)70-586-677(550,551)648-224-150(617-293(202(224)463)440-96-370-180-249(440)389-309(344)403-271(180)486)72-583-672(540,541)643-213-142(606-282(191(213)452)429-85-359-169-234(333)351-81-355-238(169)429)64-579-667(530,531)634-210-137(604-278(187(210)448)417-30-26-158(329)378-316(417)494)63-577-670(536,537)647-222-151(618-291(200(222)461)438-94-368-178-247(438)387-307(342)401-269(178)484)74-588-678(552,553)652-227-154(621-296(205(227)466)443-99-373-183-252(443)392-312(347)406-274(183)489)76-589-679(554,555)650-223-149(616-292(201(223)462)439-95-369-179-248(439)388-308(343)402-270(179)485)71-582-671(538,539)641-211-138(605-280(189(211)450)427-83-357-167-232(331)349-79-353-236(167)427)57-571-656(508,509)628-123-37-163(422-44-105(5)257(472)410-320(422)498)597-130(123)53-567-664(524,525)640-220-147(614-289(198(220)459)436-92-366-176-245(436)385-305(340)399-267(176)482)68-584-674(544,545)645-216-141(610-285(194(216)455)432-88-362-172-241(432)381-301(336)395-263(172)478)60-574-658(512,513)629-124-38-164(423-45-106(6)258(473)411-321(423)499)598-131(124)54-568-663(522,523)638-219-145(612-288(197(219)458)435-91-365-175-244(435)384-304(339)398-266(175)481)67-581-668(532,533)635-209-136(603-277(186(209)447)416-29-25-157(328)377-315(416)493)62-575-666(528,529)633-207-134(601-275(184(207)445)414-27-23-155(326)375-313(414)491)56-570-655(506,507)627-122-36-162(421-43-104(4)256(471)409-319(421)497)596-129(122)52-566-662(520,521)637-217-144(611-286(195(217)456)433-89-363-173-242(433)382-302(337)396-264(173)479)66-578-665(526,527)631-206-133(600-279(188(206)449)426-82-356-166-231(330)348-78-352-235(166)426)48-560-112-20-22-116-118(32-112)592-117-31-111(444)19-21-115(117)325(116)114-18-14-13-17-113(114)298(490)623-325/h10-32,40-47,78-99,119-154,159-165,184-230,275-297,444-467H,9,33-39,48-77,100H2,1-8H3,(H,374,468)(H,502,503)(H,504,505)(H,506,507)(H,508,509)(H,510,511)(H,512,513)(H,514,515)(H,516,517)(H,518,519)(H,520,521)(H,522,523)(H,524,525)(H,526,527)(H,528,529)(H,530,531)(H,532,533)(H,534,535)(H,536,537)(H,538,539)(H,540,541)(H,542,543)(H,544,545)(H,546,547)(H,548,549)(H,550,551)(H,552,553)(H,554,555)(H,556,557)(H,558,559)(H2,326,375,491)(H2,327,376,492)(H2,328,377,493)(H2,329,378,494)(H2,330,348,352)(H2,331,349,353)(H2,332,350,354)(H2,333,351,355)(H,407,469,495)(H,408,470,496)(H,409,471,497)(H,410,472,498)(H,411,473,499)(H,412,474,500)(H,413,475,501)(H3,334,379,393,476)(H3,335,380,394,477)(H3,336,381,395,478)(H3,337,382,396,479)(H3,338,383,397,480)(H3,339,384,398,481)(H3,340,385,399,482)(H3,341,386,400,483)(H3,342,387,401,484)(H3,343,388,402,485)(H3,344,389,403,486)(H3,345,390,404,487)(H3,346,391,405,488)(H3,347,392,406,489)")
    #longest InChi in the dataset (3915 characters) 
    print(len(encoded.tokens))
