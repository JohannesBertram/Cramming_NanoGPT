from policy import CribbagePolicy, CompositePolicy, GreedyThrower, GreedyPegger, ThrowPolicy, PegPolicy
import random
from scoring import score, greedy_throw
from deck import Card, Deck


def create_crib_tables(game):
    """Reproducing Schell tables with exploiting greedy opp strategy."""
    own_scores = {}
    opp_scores = {}
    iters1 = 10000
    count = 0
    opp_crib_throws = []
    own_crib_throws = []


    # get opp throws for both cribs
    for i in range(iters1):
        if i % 1000 == 0:
            print(i)
        deck = Deck(range(1, 14), ["S", "H", "C", "D"], 1)
        deck.shuffle()
        opp_deal = deck.deal(6)

        opp_throw = greedy_throw(game, opp_deal, 1)[1]
        opp_crib_throws.append(opp_throw)
        #print(opp_throw)        

        own_throw = greedy_throw(game, opp_deal, -1)[1]
        own_crib_throws.append(own_throw)
        #print(own_throw)

    iters2 = 1000
    for rank1 in range(1, 14):
        print(count)
        count += 1
        for rank2 in range(rank1, 14):

            own_full_total = 0
            opp_full_total = 0
            for i in range(iters2):
                # sampling from crob throws
                own_crib_throw = own_crib_throws[random.randint(0, iters1-1)]
                opp_crib_throw = opp_crib_throws[random.randint(0, iters1-1)]
                
                own_total = 0
                opp_total = 0
                
                # getting scores for all combinations
                for card_turn in game.deck()._cards:
                    own_total += score(game, [Card(rank1, "S"), Card(rank2, "C"), own_crib_throw[0], own_crib_throw[1]], card_turn, True)[0]
                    opp_total += score(game, [Card(rank1, "S"), Card(rank2, "C"), opp_crib_throw[0], opp_crib_throw[1]], card_turn, True)[0]
                own_total /= 52
                opp_total /= 52
                own_full_total += own_total
                opp_full_total += opp_total
                
            own_full_total /= iters2
            opp_full_total /= iters2
            
            own_scores[(rank1, rank2)] = own_full_total
            opp_scores[(rank1, rank2)] = opp_full_total
    print("own_crib:")
    print(own_scores)
    print("opp_crib:")
    print(opp_scores)

def sample_hand_score(game, hand, deal):
    """ Sampling hand scores with turn cards, likely not using this as too slow """
    total = 0
    for card in game.deck()._cards:
        if card not in deal:
            total += score(game, hand, card, False)[0]
    total /= 46
    return total

def sample_crib_score(game, throw, crib):
    """ Using the schell tables """
    own_scores = {(1, 1): 5.812019230769266, (1, 2): 4.673615384615375, (1, 3): 4.875192307692301, (1, 4): 5.617153846153848, (1, 5): 5.5375769230769345, (1, 6): 4.255942307692298, (1, 7): 4.233711538461524, (1, 8): 4.125730769230761, (1, 9): 3.8498461538461406, (1, 10): 3.8723461538461343, (1, 11): 4.0957499999999865, (1, 12): 3.771499999999995, (1, 13): 3.652769230769218, (2, 2): 6.255230769230801, (2, 3): 7.359173076923028, (2, 4): 5.053057692307696, (2, 5): 5.715461538461551, (2, 6): 4.3625961538461535, (2, 7): 4.432346153846139, (2, 8): 4.164519230769226, (2, 9): 4.110538461538449, (2, 10): 3.9802884615384477, (2, 11): 4.174576923076909, (2, 12): 3.923269230769228, (2, 13): 3.8949423076922987, (3, 3): 6.465461538461555, (3, 4): 5.463480769230755, (3, 5): 6.30494230769231, (3, 6): 4.2626923076923, (3, 7): 4.184461538461536, (3, 8): 4.323538461538459, (3, 9): 4.272557692307683, (3, 10): 4.032326923076908, (3, 11): 4.209230769230762, (3, 12): 4.0337692307692325, (3, 13): 3.872423076923079, (4, 4): 6.3509038461538605, (4, 5): 6.785500000000006, (4, 6): 4.498730769230755, (4, 7): 4.19213461538461, (4, 8): 4.328403846153839, (4, 9): 4.025153846153855, (4, 10): 3.880442307692301, (4, 11): 4.274019230769227, (4, 12): 3.8845192307692358, (4, 13): 3.799903846153855, (5, 5): 8.850384615384623, (5, 6): 6.845846153846157, (5, 7): 6.215961538461555, (5, 8): 5.624499999999986, (5, 9): 5.412500000000003, (5, 10): 6.747173076923084, (5, 11): 6.934192307692308, (5, 12): 6.7254230769230645, (5, 13): 6.666346153846142, (6, 6): 6.311923076923088, (6, 7): 5.334538461538472, (6, 8): 4.996173076923094, (6, 9): 5.541019230769236, (6, 10): 3.4787884615384472, (6, 11): 3.683442307692307, (6, 12): 3.331115384615385, (6, 13): 3.2914423076923027, (7, 7): 6.574403846153851, (7, 8): 6.830134615384584, (7, 9): 4.358365384615387, (7, 10): 3.5148269230768947, (7, 11): 3.7517115384615294, (7, 12): 3.4043076923076883, (7, 13): 3.325673076923065, (8, 8): 5.736576923076916, (8, 9): 4.956076923076907, (8, 10): 4.014865384615379, (8, 11): 3.7587692307692158, (8, 12): 3.5069230769230795, (8, 13): 3.329961538461534, (9, 9): 5.4969807692307615, (9, 10): 4.505403846153824, (9, 11): 4.120384615384605, (9, 12): 3.2051538461538365, (9, 13): 3.1991538461538442, (10, 10): 5.178961538461541, (10, 11): 4.653384615384593, (10, 12): 3.636884615384603, (10, 13): 3.011288461538438, (11, 11): 5.687307692307682, (11, 12): 4.691807692307677, (11, 13): 3.9844807692307747, (12, 12): 5.0363076923076955, (12, 13): 3.6139999999999985, (13, 13): 4.931480769230768}
    opp_scores = {(1, 1): 6.211826923076978, (1, 2): 4.890596153846131, (1, 3): 5.05965384615382, (1, 4): 5.755923076923072, (1, 5): 5.973884615384625, (1, 6): 4.87784615384617, (1, 7): 4.694923076923064, (1, 8): 4.77346153846154, (1, 9): 4.448557692307678, (1, 10): 4.136192307692291, (1, 11): 4.537730769230759, (1, 12): 4.056307692307677, (1, 13): 4.155749999999977, (2, 2): 6.521711538461584, (2, 3): 7.428499999999982, (2, 4): 5.131538461538479, (2, 5): 6.082057692307712, (2, 6): 4.768230769230779, (2, 7): 4.8600384615384495, (2, 8): 4.6356923076923024, (2, 9): 4.651615384615386, (2, 10): 4.364788461538452, (2, 11): 4.5235576923076835, (2, 12): 4.37576923076922, (2, 13): 4.165211538461523, (3, 3): 6.78053846153846, (3, 4): 5.812692307692275, (3, 5): 6.736057692307704, (3, 6): 4.838846153846149, (3, 7): 4.815153846153846, (3, 8): 4.878076923076915, (3, 9): 4.741115384615371, (3, 10): 4.365019230769209, (3, 11): 4.685346153846142, (3, 12): 4.310173076923069, (3, 13): 4.273615384615369, (4, 4): 6.647576923076906, (4, 5): 7.246942307692289, (4, 6): 4.9918846153846115, (4, 7): 4.842403846153841, (4, 8): 4.698576923076929, (4, 9): 4.643634615384614, (4, 10): 4.30073076923076, (4, 11): 4.483788461538461, (4, 12): 4.188365384615388, (4, 13): 4.107634615384617, (5, 5): 9.531961538461548, (5, 6): 7.527807692307717, (5, 7): 6.930230769230775, (5, 8): 6.18955769230768, (5, 9): 6.228634615384619, (5, 10): 7.279750000000013, (5, 11): 7.693538461538468, (5, 12): 7.127423076923064, (5, 13): 7.1969423076923, (6, 6): 7.079711538461529, (6, 7): 6.215692307692323, (6, 8): 5.388096153846161, (6, 9): 6.191788461538475, (6, 10): 4.130942307692301, (6, 11): 4.370076923076924, (6, 12): 4.128730769230751, (6, 13): 3.884884615384606, (7, 7): 6.89098076923077, (7, 8): 7.956442307692284, (7, 9): 5.242807692307689, (7, 10): 4.129673076923055, (7, 11): 4.354230769230758, (7, 12): 4.03767307692307, (7, 13): 4.021634615384601, (8, 8): 6.506057692307693, (8, 9): 5.8065576923076705, (8, 10): 4.702673076923067, (8, 11): 4.3402499999999815, (8, 12): 4.018865384615384, (8, 13): 3.9704230769230753, (9, 9): 6.456730769230777, (9, 10): 5.342923076923055, (9, 11): 4.8783269230769015, (9, 12): 4.006692307692282, (9, 13): 4.004499999999986, (10, 10): 5.861249999999993, (10, 11): 5.282076923076898, (10, 12): 4.088480769230767, (10, 13): 3.574942307692283, (11, 11): 6.05361538461538, (11, 12): 5.127346153846129, (11, 13): 4.431288461538466, (12, 12): 5.615807692307694, (12, 13): 4.1431346153846365, (13, 13): 5.572923076923076}

    if crib == 1:
        if throw[0].rank() < throw[1].rank():
            return own_scores[(throw[0].rank(), throw[1].rank())]
        return own_scores[(throw[1].rank(), throw[0].rank())]
    if throw[0].rank() < throw[1].rank():
        return opp_scores[(throw[0].rank(), throw[1].rank())]
    return opp_scores[(throw[1].rank(), throw[0].rank())]


def my_greedy_throw(game, deal, crib):
    """ Returns a greedy choice of which cards to throw.  The greedy choice
        is determined by the score of the four cards kept and the two cards
        thrown in isolation, without considering what the turned card
        might be or what the opponent might throw to the crib.  If multiple
        choices result in the same net score, then one is chosen randomly.

        game -- a Cribbage game
        deal -- a list of the cards dealt
        crib -- 1 for owning the crib, -1 for opponent owning the crib
    """
    def score_split(indices):

        # Run once to create crib tables
        #create_crib_tables(game)

        keep = []
        throw = []
        for i in range(len(deal)):
            if i in indices:
                throw.append(deal[i])
            else:
                keep.append(deal[i])

        keep_score = score(game, keep, None, False)[0]
        #keep_score = sample_hand_score(game, keep, deal)

        crib_score = sample_crib_score(game, throw, crib) 


        output_score = keep_score + crib * crib_score 
        
        return keep, throw, output_score

    throw_indices = game.throw_indices()
    # to randomize the order in which throws are considered to have the effect
    # of breaking ties randomly
    random.shuffle(throw_indices)

    # pick the (keep, throw, score) triple with the highest score
    return max(map(lambda i: score_split(i), throw_indices), key=lambda t: t[2])

class MyGreedyThrower(ThrowPolicy):
    """ A greedy policy for keep/throw in cribbage.  The greedy decision is
        based only on the score obtained by the cards kept and thrown, without
        consideration for how they might interact with the turned card or
        cards thrown by the opponent.
    """
    
    def __init__(self, game):
        """ Creates a greedy keep/throw policy for the given game.

            game -- a cribbage Game
        """
        super().__init__(game)


    def keep(self, hand, scores, am_dealer):
        """ Selects the cards to keep to maximize the net score for those cards
            and the cards in the crib.  Points in the crib count toward the
            total if this policy is the dealer and against the total otherwise.

            hand -- a list of cards
            scores -- the current scores, with this policy's score first
            am_dealer -- a boolean flag indicating whether the crib
                         belongs to this policy
        """
        keep, throw, net_score = my_greedy_throw(self._game, hand, 1 if am_dealer else -1)
        return keep, throw
    
    
def five_or_ten_score(card):
    if card.rank() == 5:
        return -1
    if card.rank() == 10:
        return -0.5
    return 0

def card_in_pair_score(card, hand):
    rank = card.rank()
    num_same_rank = 0
    for card in hand:
        if card.rank() == rank:
            num_same_rank += 1
    if num_same_rank == 2 and rank != 5: # pair trap with 5 not working as gready opponent will more likely just play 10 instead
        return 0.5
    return 0

def card_low(card):
    if card.rank() < 5:
        return 0.1
    return 0

def dont_hit_21(card, total_points):
    if total_points + card.rank() == 21:
        return -0.5
    return 0

def get_close_to_31(card, total_points):
    card_val = card.rank() if card.rank() < 10 else 10
    if total_points + card_val > 21:
        return (5 - (31 - total_points - card_val)) / 10
    return 0


class MyGreedyPegger(PegPolicy):
    """ A cribbage pegging policy that plays the card that maximizes the
        points earned on the current play.
    """

    def __init__(self, game):
        """ Creates a greedy pegging policy for the given game.

            game -- a cribbage Game
        """
        super().__init__(game)


    def peg(self, cards, history, turn, scores, am_dealer):
        """ Returns the card that maximizes the points earned on the next
            play.  Ties are broken uniformly randomly.

            cards -- a list of cards
            history -- the pegging history up to the point to decide what to play
            turn -- the cut card
            scores -- the current scores, with this policy's score first
            am_dealer -- a boolean flag indicating whether the crib
                         belongs to this policy
        """
        # shuffle cards to effectively break ties randomly
        random.shuffle(cards)

        best_card = None
        best_score = None
        for card in cards:
            score = history.score(self._game, card, 0 if am_dealer else 1)

            if history.is_start_round():
                score += five_or_ten_score(card) # 5 or 10 start allows 15 for opp
                score += card_in_pair_score(card, cards) # trap opp with pair
                score += card_low(card) # break ties to play low cards preventing 15 from opp
            elif score is not None:
                score += dont_hit_21(card, history.total_points()) # dont hit 21 as opponent often has 10 card to reach 31
                score += get_close_to_31(card, history.total_points()) # get as close to 31 as possible to get 1 point for last card
                
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_card = card
        return best_card

class MyPolicy(CribbagePolicy):
    def __init__(self, game):
        self._policy = CompositePolicy(game, MyGreedyThrower(game), MyGreedyPegger(game))

        
    def keep(self, hand, scores, am_dealer):
        return self._policy.keep(hand, scores, am_dealer)


    def peg(self, cards, history, turn, scores, am_dealer):
        return self._policy.peg(cards, history, turn, scores, am_dealer)



    

                                    
