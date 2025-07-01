import pygame
import sys
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.metrics.pairwise  import cosine_similarity

# sample bollywood movie dataset (title,description)
movies = [
    ("jab we met", "comedy self love romance drama"),
    ("yeh jawaani hai deewani", "friendship adventure romance self discovery"),
    ("tamasha", "drama identity romance self love"),
    ("wake up sid", "coming of age urban drama self growth"),
    ("zindagi na milegi dobara", "road trip friendship self discovery humor"),
    ("rockstar", "romance passion music heartbreak drama"),
    ("barfi!", "quirky romance comedy emotional"),
    ("dil chahta hai", "friendship youth romance drama"),
    ("queen", "solo travel self love empowerment comedy"),
    ("dear zindagi", "mental health self discovery drama coming of age"),
    ("love aaj kal", "romance modern vs traditional love drama"),
    ("city lights serenade", "romance city life humor self discovery"),
    ("wanderlust whispers", "travel friendship self discovery romance"),
    ("midnight diaries", "urban drama introspection love"),
    ("echoes of youth", "coming of age drama friendship nostalgia"),
    ("paper planes", "coming of age youth self growth"),
    ("heartbeat highway", "music romance adventure self identity"),
    ("canvas of dreams", "artistic journey identity self expression"),
    ("soulful sunsets", "romance self love restored faith"),
    ("tangled timelines", "drama identity memory love"),
    ("crossroads café", "slice of life comedy romance drama"),
    ("mirror reflections", "introspection self love drama"),
    ("pilgrim hearts", "spiritual journey friendship self discovery"),
    ("river of voices", "spiritual journey self awakening"),
    ("whispering valleys", "rural adventure self discovery"),
    ("fragments of us", "identity drama self acceptance romance"),
    ("starlit promises", "romance destiny self belief"),
    ("silent conversations", "urban drama love self awareness"),
    ("melody of memories", "nostalgia romance heartfelt"),
    ("journey to joy", "self improvement comedy heartwarming"),
    ("monsoon melodies", "music love self healing nostalgia"),
    ("rhythm of rain", "music passion romance heartbreak"),
    ("city of strangers", "urban drama connection loneliness"),
    ("gilded promises", "period romance drama self rebirth"),
    ("ocean of dreams", "romance journey self belief"),
    ("vintage voyage", "nostalgia travel self rediscovery"),
    ("daffodil diaries", "coming of age romance self expression"),
    ("umbrella for two", "romantic comedy city life"),
    ("sunrise pact", "friendship promise self growth"),
    ("paper trails", "urban drama ambition self discovery"),
    ("lost & found souls", "drama redemption love self healing"),
    ("wanderer’s waltz", "dance romance journey"),
    ("stargazing strangers", "romance serendipity identity"),
    ("zodiac promise", "destiny romance self belief"),
    ("mirror maze", "identity thriller drama self conflict"),
    ("diary of dawn", "self reflection coming of age"),
    ("chasing monsoon", "friendship road trip romance self discovery"),
    ("inked in love", "artistic romance self identity"),
    ("tide of truths", "family drama self revelation"),
    ("serenade of silence", "music introspection self healing"),
    ("backroad banter", "road trip comedy friendship"),
    ("chai & choices", "slice of life drama decision self love"),
    ("drifters’ dance", "road trip romance self discovery"),
    ("saffron skies", "spiritual romance self healing"),
    ("harbor of hope", "redemption drama self growth"),
    ("city pulse", "urban adventure self discovery"),
    ("flashes of feeling", "emotional romance friendship drama"),
    ("wildflower dreams", "self love empowerment drama"),
    ("labyrinth of laughter", "comedy ensemble friendship"),
    ("young & yearning", "coming of age romance identity"),
    ("riverbank rhapsody", "music romance soothing"),
    ("broken strings", "music heartbreak self recovery"),
    ("xenial strangers", "serendipity romance friendship"),
    ("nocturnal novellas", "nightlife drama self exploration"),
    ("mosaic souls", "drama identity love diversity"),
    ("grayscale hearts", "drama healing emotional connection"),
    ("pedal to passion", "sports drama romance self determination"),
    ("whims of fate", "romance chance mystery drama"),
    ("ink & innocence", "coming of age romance art"),
    ("tales from terrace", "urban drama friendship romance"),
    ("nomad’s notebook", "travel diary self discovery"),
    ("dawning desires", "new beginnings romance self love"),
    ("echo park stories", "urban drama growth love"),
    ("mirrorlake meditations", "introspective drama mental health"),
    ("moonset memoirs", "melancholy romance self reflection"),
    ("aurora aspirations", "cosmic romance self hope"),
    ("breadcrumbs to bliss", "romantic comedy self discovery"),
    ("wings of wonder", "fantasy drama hope self discovery"),
    ("caramel sunrise", "feel good romance self love"),
    ("solstice song", "music drama romance journey"),
    ("vow under stars", "romance drama destiny commitment"),
    ("meadow of minds", "philosophy drama self discovery"),
    ("garden of ghosts", "mystery drama self healing"),
    ("embers & echoes", "nostalgia drama self reflection"),
    ("letters in rain", "romance secrets heartbreak drama"),
    ("kaleidoscope skies", "romance art drama aesthetic"),
    ("fables of fall", "seasonal romance drama nostalgia"),
    ("painter’s promise", "art romance self fulfillment"),
    ("emberlight", "emotional drama resilience self discovery"),
    ("soul on safari", "travel adventure self discovery romance"),
    ("zest of life", "comedy drama self celebration"),
    ("harvest of hearts", "rural romance drama community"),
    ("quill & quiver", "historical drama self love"),
    ("year of yes", "empowerment self love comedy"),
    ("zealous zenith", "ambition self achievement drama"),
    ("midday mirages", "desert romance self testing"),
    ("altar of aspirations", "motivational drama self growth"),
    ("fleeting footprints", "travel memoir self reflection"),
    ("veranda views", "family drama nostalgia self reflection"),
    ("dreamers anonymous", "urban comedy friendship goals"),
    ("parlor philosophies", "slice of life comedy drama")
]

# extract title and descriptions
titles = [title.lower() for title,desc in movies]
description = [desc for title,desc in movies]

# vectorize the description
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(description)

#compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# recommendation function
def recommend(title,top_n=5):
    title = title.lower()
    if title not in titles:
        return["Movie not found in database."]
    idx = titles.index(title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores,key = lambda x:x[1], reverse = True)
    
    recommendations = []
    for i, score in sim_scores[1:top_n +1]:
        recommendations.append(f"{movies[i][0]} (similarity:{score:.2f})")
    return recommendations

#pygame setup
pygame.init()
WIDTH,HEIGHT = 800,600
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Bollywood Movie Recommendation System")

WHITE = (255,255,255)
LIGHT_GRAY = (240,240,240)
GRAY = (200,200,200)
DARK_GRAY = (50,50,50)
BLACK = (0,0,0)
BLUE = (100,149,237)
DARK_BLUE = (70,130,180)
BG_TOP = (255,200,200)
BG_BOTTOM = (200,220,255)

font = pygame.font.SysFont("Arial",24)
big_font = pygame.font.SysFont("Arial",36,bold = True)
title_font = pygame.font.SysFont("Arial",28,bold = True)

input_box = pygame.Rect(50,80,700,45)
input_text = ''
active = False

button_Rect = pygame.Rect(330,140,140,45)

output_box = pygame.Rect(50,220,700,250)
recommendations = []

def draw_rounded_rect(surface,color,rect,radius = 10):
    """Draw a rounded rectangle."""
    pygame.draw.rect(surface,color,rect,border_radius = radius)

def draw_gradient_background(surface,top_color,bottom_color):
    """Draw a vertical gradient background."""
    for y in range(HEIGHT):
        blend = y/HEIGHT
        r = int(top_color[0]*(1-blend)+bottom_color[0]*blend)
        g = int(top_color[1]*(1-blend)+bottom_color[1]*blend)        
        b = int(top_color[2]*(1-blend)+bottom_color[2]*blend)
        pygame.draw.line(surface, (r,g,b), (0,y),(WIDTH,y))

running = True
while running:
    draw_gradient_background(screen,BG_TOP,BG_BOTTOM)

    header = title_font.render("Bollywood Movie Recommendation System",True,DARK_GRAY)
    screen.blit(header,(WIDTH//2-header.get_width()//2,20))

    label = font.render("Enter Movie Title:",True,BLACK)
    screen.blit(label,(50,50))

    draw_rounded_rect(screen,WHITE,input_box,radius = 8)
    pygame.draw.rect(screen,BLUE if active else GRAY,input_box,2, border_radius = 8)
    txt_surface = font.render(input_text, True,BLACK)
    screen.blit(txt_surface,(input_box.x + 10, input_box.y +10))

    draw_rounded_rect(screen,DARK_BLUE,button_Rect,radius = 8)
    button_text = font.render("Recommend", True,WHITE)
    screen.blit(button_text,(button_Rect.x + 20, button_Rect.y +10))

    draw_rounded_rect(screen,WHITE,output_box,radius = 10)
    pygame.draw.rect(screen,DARK_BLUE,output_box,2, border_radius = 10)

    y = output_box.y+20
    if recommendations:
        result_label = font.render("Top Recommendations:",True,BLACK)
        screen.blit(result_label,(output_box.x+20,y))
        y+= 40
        for rec in recommendations:
            bullet_x = output_box.x +30
            bullet_y = y+10
            pygame.draw.circle(screen,DARK_BLUE,(bullet_x,bullet_y),5)
            rec_text = font.render(rec,True,DARK_GRAY)
            screen.blit(rec_text,(bullet_x+20,y))
            y+=35
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if input_box.collidepoint(event.pos):
                active = True
            else:
                active = False

            if button_Rect.collidepoint(event.pos):
                if input_text.strip():
                    recommendations = recommend(input_text.strip())
                else:
                    recommendations = ["please enter a movie title"]

        elif event.type == pygame.KEYDOWN:
            if active:
                if event.key == pygame.K_RETURN:
                    if input_text.strip():
                        recommendations = recommend(input_text.strip())
                    else:
                        recommendations = ["please enter a movie title"]
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode

    pygame.display.flip()

pygame.quit()
sys.exit()