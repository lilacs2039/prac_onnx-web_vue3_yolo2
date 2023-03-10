import { createRouter, createWebHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import YoloView from "../views/YoloView.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/Home",
      name: "home",
      component: HomeView,
    },
    {
      path: "/about",
      name: "about",
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("../views/AboutView.vue"),
    },
    {
      path: "/",
      name: "yolo",
      // component: () => import("../views/Yolo.vue"),
      component: YoloView,
    },
  ],
});

export default router;
