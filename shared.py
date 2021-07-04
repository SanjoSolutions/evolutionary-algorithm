from road.road import Road


def car_index_to_embedding(car_index):
    embedding = [0] * Road.NUMBER_OF_ROADS
    embedding[car_index] = 1
    return tuple(embedding)
