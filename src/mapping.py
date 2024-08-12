# GROUPS_TEMPLATE = [
#     {
#         'id': 0,
#         'name': '',
#         'topics': [],
#     },
#     ...
# ]

# (1)~вопросы пользователя по политике безопасности, 
# (2)~сторонние веб-сайты, 
# (3)~особая аудитория, 
# (4)~защита персональных данных, 
# (5)~сбор персональных данных, 
# (6)~сбор данных, позволяющих отслеживать поведения пользователя на веб-сайте, 
# (7)~распространение персональных (передача третьим лицам), 
# (8)~хранение персональных данных, 
# (9)~персонализация и маркетинг, 
# (10)~изменение политики безопасности.

GROUPS_WITH_BETTER_PREPROCESS_EN = [
    {
        'id': 0,
        'name': 'Вопросы пользователя по политике безопасности',
        'topics': [0, ],
    },
    {
        'id': 1,
        'name': 'Сторонние веб-сайты',
        'topics': [2, ],
    },
    {
        'id': 2,
        'name': 'Особая аудитория',
        'topics': [3, 6, ],
    },
    {
        'id': 3,
        'name': 'Защита персональных данных',
        'topics': [4, ],
    },
    {
        'id': 4,
        'name': 'Сбор персональных данных',
        'topics': [5, 9, 10, 1, 12, 14, 17, 21, 22, ],
    },
    {
        'id': 5,
        'name': 'Сбор данных, позволяющих отслеживать поведения пользователя на веб-сайте',
        'topics': [18, ],
    },
    {
        'id': 6,
        'name': 'Передача персональных данных третьим лицам',
        'topics': [7, 11, 15, ],
    },
    {
        'id': 7,
        'name': 'Хранение персональных данных',
        'topics': [8, ],
    },
    {
        'id': 8,
        'name': 'Персонализация и маркетинг',
        'topics': [13, 20, ],
    },
    {
        'id': 9,
        'name': 'Изменение политики безопасности',
        'topics': [16, 19, ],
    },
]

# (1) термины и определения политики, 
# (2) сбор трекинговых персональных данных, 
# (3) сбор, обработка и хранение персональных данных, 
# (4) передача третьим лицам и уничтожение персональных данных, 
# (5) изменение персональных данных, 
# (6) разрешение споров, 
# (7) уведомление, маркетинг и персонализация, 
# (8) цели обработки персональных данных, 
# (9) защита персональных данных, 
# (10) правовые основания обработки, 
# (11) обновление политики безопасности, 
# (12) согласие пользователя на обработку персональных. 

GROUPS_WITH_BETTER_PREPROCESS_RU = [
    {
        'id': 0,
        'name': 'Термины и определения политики',
        'topics': [5, 8, 10, 11, 34 ],
    },
    {
        'id': 1,
        'name': 'Сбор трекинговых персональных данных',
        'topics': [2, 13, 18, 26, 30, 44  ],
    },
    {
        'id': 2,
        'name': 'Сбор, обработка и хранение персональных данных',
        'topics': [0, 1, 3, 12, 15, 22, 28, 32, 37, 41,  ],
    },
    {
        'id': 3,
        'name': 'Передача третьим лицам и уничтожение персональных данных',
        'topics': [19, 39, 40, 24,  ],
    },
    {
        'id': 4,
        'name': 'Изменение персональных данных',
        'topics': [17, 36 ],
    },
    {
        'id': 5,
        'name': 'Разрешение споров',
        'topics': [6,  ],
    },
    {
        'id': 6,
        'name': 'Уведомление, маркетинг и персонализация',
        'topics': [16, 20,  ],
    },
    {
        'id': 7,
        'name': 'Цели обработки персональных данных',
        'topics': [4, 25, 29, 38,  ],
    },
    {
        'id': 8,
        'name': 'Защита персональных данных',
        'topics': [31, 42,  ],
    },
    {
        'id': 9,
        'name': 'Правовые основания обработки персональных данных',
        'topics': [14, 21, 23, 33, 35,  ],
    },
    {
        'id': 10,
        'name': 'Обновление политики безопасности',
        'topics': [43,  ],
    },
    {
        'id': 11,
        'name': 'Согласие пользователя на обработку персональных данных',
        'topics': [9, 27,  ],
    },
]
