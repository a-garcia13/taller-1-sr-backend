from services import recomendation

rs = recomendation.RecomendationService()

tracks = rs.make_prediction("user_000001")

print(tracks)